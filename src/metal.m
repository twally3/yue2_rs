// Native Apple matrix kernels and bounded resource reuse for both GPU families.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

@interface Yue2Context : NSObject
@property(nonatomic,strong) id<MTLBuffer> data;
@property(nonatomic) size_t length,capacity;
@end
@implementation Yue2Context
@end
static NSMutableDictionary<NSString *, Yue2Context *> *arContexts;
static size_t arContextBytes;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static NSLock *lock;
static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *kernels;
static NSMutableDictionary<NSString *, MPSMatrix *> *weights;
static id<MTLBuffer> buffers[52];
static NSMutableDictionary<NSString *, id<MTLBuffer>> *prefixes;
static size_t prefixBytes, prefixBudget;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *vectors;
static size_t vectorBytes, vectorBudget;
static BOOL vectorMode;
static size_t weightBytes, weightBudget;
uint64_t yue2_physical_memory(void) { return NSProcessInfo.processInfo.physicalMemory; }

static void initialize(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
        if (device && MPSSupportsMTLDevice(device)) {
            queue = [device newCommandQueue];
            lock = [NSLock new]; kernels = [NSMutableDictionary new];
            weights = [NSMutableDictionary new]; prefixes = [NSMutableDictionary new];
            vectors = [NSMutableDictionary new];arContexts=[NSMutableDictionary new];
            vectorBudget=MIN(yue2_physical_memory()/4,device.recommendedMaxWorkingSetSize*3/4);
            prefixBudget = MIN(device.hasUnifiedMemory ? yue2_physical_memory()/24
                : device.recommendedMaxWorkingSetSize/10, (size_t)384<<20);
            size_t automatic = device.hasUnifiedMemory
                ? MIN(yue2_physical_memory()/12, (size_t)2<<30)
                : MIN(device.recommendedMaxWorkingSetSize/5, (size_t)768<<20);
            NSString *setting = NSProcessInfo.processInfo.environment[@"YUE2_GPU_CACHE_MB"];
            weightBudget=automatic;
            if (setting) {
                const char *start=setting.UTF8String;char *end=NULL;errno=0;
                unsigned long long mb=strtoull(start,&end,10);
                if (!errno && end!=start && *end=='\0' && *start!='-'
                    && mb<=SIZE_MAX/(1<<20)) {
                    weightBudget=(size_t)mb*(1<<20);
                    vectorBudget=MIN(weightBudget,MIN(yue2_physical_memory()/3,
                        device.recommendedMaxWorkingSetSize*3/4));
                }
            }
            weightBudget=MIN(weightBudget,MIN(yue2_physical_memory()/3,
                device.recommendedMaxWorkingSetSize/2));
        } else { device = nil; }
    });
}
int yue2_metal_available(void) { initialize(); return queue != nil; }
const char *yue2_metal_name(void) { initialize(); return device ? device.name.UTF8String : "CPU"; }
void yue2_metal_clear_weights(void) {
    initialize(); if (!queue) return;
    [lock lock]; [weights removeAllObjects]; weightBytes=0; [prefixes removeAllObjects]; prefixBytes=0; [vectors removeAllObjects];vectorBytes=0;vectorMode=NO;[arContexts removeAllObjects];arContextBytes=0; [lock unlock];
}
void yue2_metal_release_model(void) {
    initialize();if (!queue) return;
    [lock lock];
    [weights removeAllObjects];weightBytes=0;[prefixes removeAllObjects];prefixBytes=0;[vectors removeAllObjects];vectorBytes=0;vectorMode=NO;[arContexts removeAllObjects];arContextBytes=0;
    for (size_t slot=19;slot<52;++slot) buffers[slot]=nil;
    [lock unlock];
}
static MTLResourceOptions hostStorage(void) {
    return device.hasUnifiedMemory ? MTLResourceStorageModeShared : MTLResourceStorageModeManaged;
}
static id<MTLBuffer> buffer(size_t slot, size_t length, MTLResourceOptions options) {
    if (length > device.maxBufferLength) return nil;
    id<MTLBuffer> old = buffers[slot];
    if (!old || old.length < length || old.storageMode != (options & MTLResourceStorageModeMask)>>MTLResourceStorageModeShift) {
        size_t capacity = MIN(device.maxBufferLength, (length + length/8 + 65535) & ~(size_t)65535);
        old = [device newBufferWithLength:capacity options:options];
        buffers[slot] = old;
    }
    return old;
}
static MPSMatrix *view(id<MTLBuffer> data, size_t rows, size_t cols, size_t stride) {
    if (!data) return nil;
    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:cols rowBytes:stride*sizeof(float) dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:data descriptor:descriptor];
}
static MPSMatrix *matrix(const float *data, size_t rows, size_t cols,
                         size_t stride, BOOL copy, size_t slot) {
    size_t packed = (cols + 15) & ~(size_t)15;
    id<MTLBuffer> storage = buffer(slot, rows*packed*sizeof(float), hostStorage());
    if (!storage) return nil;
    if (copy) {
        float *target = storage.contents;
        for (size_t r=0;r<rows;++r) {
            memcpy(target+r*packed,data+r*stride,cols*sizeof(float));
            if (packed>cols) memset(target+r*packed+cols,0,(packed-cols)*sizeof(float));
        }
        if (storage.storageMode==MTLStorageModeManaged)
            [storage didModifyRange:NSMakeRange(0,rows*packed*sizeof(float))];
    }
    return view(storage,rows,cols,packed);
}
static MPSMatrixMultiplication *kernel(size_t m,size_t n,size_t k,BOOL transpose,float beta) {
    NSString *key=[NSString stringWithFormat:@"%zu/%zu/%zu/%d/%g",m,n,k,transpose,beta];
    MPSMatrixMultiplication *op=kernels[key];
    if (!op) {
        op=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO
            transposeRight:transpose resultRows:m resultColumns:n interiorColumns:k alpha:1 beta:beta];
        if (kernels.count>256) [kernels removeAllObjects];
        kernels[key]=op;
    }
    op.batchStart=0; op.batchSize=1;
    return op;
}
static BOOL finish(id<MTLCommandBuffer> command,MPSMatrix *result,float *c,size_t m,size_t n,size_t ldc) {
    if (result.data.storageMode==MTLStorageModeManaged) {
        id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
        if (!blit) return NO;
        [blit synchronizeResource:result.data]; [blit endEncoding];
    }
    [command commit]; [command waitUntilCompleted];
    if (command.status!=MTLCommandBufferStatusCompleted) return NO;
    const float *source=result.data.contents;
    size_t stride=result.rowBytes/sizeof(float);
    for (size_t r=0;r<m;++r) memcpy(c+r*ldc,source+r*stride,n*sizeof(float));
    return YES;
}
int yue2_metal_gemm(const float *a,const float *b,float *c,
                   size_t m,size_t n,size_t k,int transpose_b,
                   size_t lda,size_t ldb,size_t ldc,float beta) {
    initialize(); if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        MPSMatrix *left=matrix(a,m,k,lda,YES,0);
        MPSMatrix *right=matrix(b,transpose_b?n:k,transpose_b?k:n,ldb,YES,1);
        MPSMatrix *result=matrix(c,m,n,ldc,beta!=0,2);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        BOOL ok=left&&right&&result&&command;
        if (ok) {
            [kernel(m,n,k,transpose_b!=0,beta) encodeToCommandBuffer:command
                leftMatrix:left rightMatrix:right resultMatrix:result];
            ok=finish(command,result,c,m,n,ldc);
        }
        [lock unlock]; return ok;
    }
}
static id<MTLComputePipelineState> packQ,packKV,softmax,unpack,rms,rotary,point,merge,weightConvert,weightVector,cachedAttention,appendContext;
static BOOL attentionInitialize(void);
static void dispatch(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,size_t count);
typedef struct { uint32_t rows,cols,stride,dtype,bf16; } WeightShape;
static MPSMatrix *converted(const uint8_t *raw,int dtype,BOOL rounded,
                           size_t mapping,size_t n,size_t k,id<MTLCommandBuffer> command,
                           NSString **keyOut,BOOL *admit,size_t slot) {
    NSString *key=[NSString stringWithFormat:@"%zu/%p/%zu/%zu/%d/%d",mapping,raw,n,k,dtype,rounded];
    *keyOut=key;
    MPSMatrix *cached=weights[key];
    if (cached) { *admit=NO; return cached; }
    size_t stride=(k+15)&~(size_t)15, length=n*stride*sizeof(float);
    if (!attentionInitialize()||!weightConvert) return nil;
    size_t rawLength=n*k*(dtype==1 ? 4 : 2);
    id<MTLBuffer> staging=buffer(slot,rawLength,hostStorage());
    id<MTLBuffer> source=buffer(slot==3 ? 46 : 39+slot-32,rawLength,MTLResourceStorageModePrivate);
    id<MTLBuffer> gpu=[device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    if (!staging||!source||!gpu) return nil;
    memcpy(staging.contents,raw,rawLength);
    if (staging.storageMode==MTLStorageModeManaged) [staging didModifyRange:NSMakeRange(0,rawLength)];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    if (!blit) return nil;
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:source destinationOffset:0 size:rawLength];
    [blit endEncoding];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) return nil;
    WeightShape shape={(uint32_t)n,(uint32_t)k,(uint32_t)stride,(uint32_t)dtype,(uint32_t)rounded};
    [encoder setBuffer:source offset:0 atIndex:0];[encoder setBuffer:gpu offset:0 atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    dispatch(encoder,weightConvert,n*stride);[encoder endEncoding];
    *admit=length<=weightBudget-weightBytes;
    return view(gpu,n,k,stride);
}
int yue2_metal_linear(const float *a,const uint8_t *raw,size_t mapping,
                      int dtype,int rounded,float *c,size_t m,size_t n,size_t k) {
    initialize(); if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        if (vectorMode) { [vectors removeAllObjects];vectorBytes=0;vectorMode=NO;[arContexts removeAllObjects];arContextBytes=0; }
        id<MTLCommandBuffer> command=[queue commandBuffer];
        if (!command) { [lock unlock];return 0; }
        MPSMatrix *left=matrix(a,m,k,k,YES,0);
        MPSMatrix *result=matrix(c,m,n,n,NO,2);
        NSString *key; BOOL admit=NO;
        MPSMatrix *right=left&&result ? converted(raw,dtype,rounded,mapping,n,k,command,&key,&admit,3) : nil;
        BOOL ok=left&&right&&result;
        if (ok) {
            [kernel(m,n,k,YES,0) encodeToCommandBuffer:command
                leftMatrix:left rightMatrix:right resultMatrix:result];
            ok=finish(command,result,c,m,n,n);
            if (ok&&admit) {weights[key]=right;weightBytes+=right.data.length;}
        }
        [lock unlock];return ok;
    }
}

#include "yue2_shader.inc"

static BOOL attentionInitialized;
static BOOL attentionInitialize(void) {
    if (!attentionInitialized) {
        attentionInitialized=YES;
        MTLCompileOptions *options=[MTLCompileOptions new];
        options.fastMathEnabled=NO; options.languageVersion=MTLLanguageVersion2_2;
        NSError *error=nil;
        id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:yue2_shader]
            options:options error:&error];
        if (library) {
            packQ=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack_queries"] error:&error];
            packKV=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack_keys_values"] error:&error];
            softmax=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_softmax"] error:&error];
            unpack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"unpack_attention"] error:&error];
            rms=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rms_norm"] error:&error];
            rotary=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"normalize_rotary"] error:&error];
            point=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pointwise"] error:&error];
            merge=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"merge_keys_values"] error:&error];
            weightConvert=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"convert_weight"] error:&error];
            weightVector=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"weight_vector"] error:&error];
            cachedAttention=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"cached_attention"] error:&error];
            appendContext=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"append_context"] error:&error];
        }
        if (!packQ||!packKV||!softmax||!unpack)
            fprintf(stderr,"YuE2 GPU attention initialization failed: %s\n",error.localizedDescription.UTF8String);
    }
    return packQ&&packKV&&softmax&&unpack&&softmax.maxTotalThreadsPerThreadgroup>=256;
}
typedef struct {
    uint32_t rows,keys,heads,kv_heads,dim,groups,start,tile,score_stride;
    int32_t causal;
    float scale;
} AttentionShape;
static MPSMatrix *batchView(id<MTLBuffer> data,size_t rows,size_t cols,size_t stride,size_t batches) {
    if (!data) return nil;
    MPSMatrixDescriptor *descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
        matrices:batches rowBytes:stride*sizeof(float) matrixBytes:rows*stride*sizeof(float)
        dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:data descriptor:descriptor];
}
static void dispatch(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,size_t count) {
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static BOOL encodeAttention(id<MTLCommandBuffer> command,id<MTLBuffer> q,
    id<MTLBuffer> pk,id<MTLBuffer> pv,id<MTLBuffer> out,AttentionShape shape) {
    size_t rows=shape.rows,keys=shape.keys,heads=shape.heads,kv=shape.kv_heads,dim=shape.dim;
    size_t groups=shape.groups,maxTile=MIN(rows,(size_t)256),stride=shape.score_stride;
    id<MTLBuffer> pq=buffer(7,heads*maxTile*dim*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> scores=buffer(10,heads*maxTile*stride*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> product=buffer(11,heads*maxTile*dim*4,MTLResourceStorageModePrivate);
    if (!pq||!scores||!product) return NO;
    id<MTLComputeCommandEncoder> encoder;
        for (size_t start=0;start<rows;start+=maxTile) {
            size_t tile=MIN(maxTile,rows-start), packedRows=groups*tile;
            shape.start=(uint32_t)start;shape.tile=(uint32_t)tile;
            encoder=[command computeCommandEncoder];
            if (!encoder) { return NO; }
            [encoder setBuffer:q offset:0 atIndex:0]; [encoder setBuffer:pq offset:0 atIndex:1];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
            dispatch(encoder,packQ,heads*tile*dim); [encoder endEncoding];
            MPSMatrix *query=batchView(pq,packedRows,dim,dim,kv);
            MPSMatrix *key=batchView(pk,keys,dim,dim,kv);
            MPSMatrix *score=batchView(scores,packedRows,keys,stride,kv);
            MPSMatrixMultiplication *dot=kernel(packedRows,keys,dim,YES,0);
            dot.batchSize=kv;
            [dot encodeToCommandBuffer:command leftMatrix:query rightMatrix:key resultMatrix:score];
            encoder=[command computeCommandEncoder];
            if (!encoder) { return NO; }
            [encoder setComputePipelineState:softmax];
            [encoder setBuffer:scores offset:0 atIndex:0];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(heads*tile,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];
            MPSMatrix *value=batchView(pv,keys,dim,dim,kv);
            MPSMatrix *weighted=batchView(product,packedRows,dim,dim,kv);
            MPSMatrixMultiplication *sum=kernel(packedRows,dim,keys,NO,0);
            sum.batchSize=kv;
            [sum encodeToCommandBuffer:command leftMatrix:score rightMatrix:value resultMatrix:weighted];
            encoder=[command computeCommandEncoder];
            if (!encoder) { return NO; }
            [encoder setBuffer:product offset:0 atIndex:0]; [encoder setBuffer:out offset:0 atIndex:1];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
            dispatch(encoder,unpack,heads*tile*dim); [encoder endEncoding];
        }
    return YES;
}
int yue2_metal_attention(const float *q,const float *k,const float *v,float *out,
                         size_t rows,size_t keys,size_t heads,size_t kv,size_t dim,
                         int64_t causal,float scale) {
    initialize(); if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        if (!attentionInitialize()) { [lock unlock];return 0; }
        size_t groups=heads/kv, maxTile=MIN(rows,(size_t)256), stride=(keys+15)&~(size_t)15;
        MPSMatrix *qs=matrix(q,rows,heads*dim,heads*dim,YES,4);
        MPSMatrix *ks=matrix(k,keys,kv*dim,kv*dim,YES,5);
        MPSMatrix *vs=matrix(v,keys,kv*dim,kv*dim,YES,6);
        id<MTLBuffer> pq=buffer(7,heads*maxTile*dim*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> pk=buffer(8,kv*keys*dim*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> pv=buffer(9,kv*keys*dim*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> scores=buffer(10,heads*maxTile*stride*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> product=buffer(11,heads*maxTile*dim*4,MTLResourceStorageModePrivate);
        MPSMatrix *result=matrix(out,rows,heads*dim,heads*dim,NO,12);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        if (!qs||!ks||!vs||!pq||!pk||!pv||!scores||!product||!result||!command) { [lock unlock];return 0; }
        AttentionShape shape={(uint32_t)rows,(uint32_t)keys,(uint32_t)heads,(uint32_t)kv,
            (uint32_t)dim,(uint32_t)groups,0,(uint32_t)maxTile,(uint32_t)stride,(int32_t)causal,scale};
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        if (!encoder) { [lock unlock];return 0; }
        [encoder setBuffer:ks.data offset:0 atIndex:0]; [encoder setBuffer:vs.data offset:0 atIndex:1];
        [encoder setBuffer:pk offset:0 atIndex:2]; [encoder setBuffer:pv offset:0 atIndex:3];
        [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
        dispatch(encoder,packKV,kv*keys*dim); [encoder endEncoding];
        if (!encodeAttention(command,qs.data,pk,pv,result.data,shape)) { [lock unlock];return 0; }
        BOOL ok=finish(command,result,out,rows,heads*dim,heads*dim);
        [lock unlock];return ok;
    }
}

typedef struct {
    const uint8_t *raw; size_t mapping,rows,cols; int32_t dtype,rounded;
} GpuWeight;
typedef struct {
    GpuWeight weights[7]; const float *norms[4]; const float *key,*value;
} GpuLayer;
typedef struct { uint32_t rows,cols; float eps; } NormShape;
typedef struct { uint32_t rows,heads,dim; float eps; } HeadShape;
typedef struct { uint32_t count,mode; } PointShape;
typedef struct { uint32_t rows,prefix,kv,dim; } MergeShape;
static BOOL encodeNorm(id<MTLCommandBuffer> command,id<MTLBuffer> input,
    id<MTLBuffer> gamma,size_t offset,id<MTLBuffer> output,size_t rows,size_t cols,float eps) {
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) return NO;
    NormShape shape={(uint32_t)rows,(uint32_t)cols,eps};
    [encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:gamma offset:offset atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder setComputePipelineState:rms];
    [encoder dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding];return YES;
}
static BOOL encodeRotary(id<MTLCommandBuffer> command,id<MTLBuffer> input,
    id<MTLBuffer> gamma,size_t offset,id<MTLBuffer> cosine,id<MTLBuffer> sine,
    size_t rows,size_t heads,size_t dim,float eps) {
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) return NO;
    HeadShape shape={(uint32_t)rows,(uint32_t)heads,(uint32_t)dim,eps};
    [encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:gamma offset:offset atIndex:1];
    [encoder setBuffer:cosine offset:0 atIndex:2];[encoder setBuffer:sine offset:0 atIndex:3];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
    [encoder setComputePipelineState:rotary];
    [encoder dispatchThreadgroups:MTLSizeMake(rows*heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding];return YES;
}
static BOOL encodePoint(id<MTLCommandBuffer> command,id<MTLBuffer> a,id<MTLBuffer> b,
    size_t count,size_t mode) {
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) return NO;
    PointShape shape={(uint32_t)count,(uint32_t)mode};
    [encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    dispatch(encoder,point,count);[encoder endEncoding];return YES;
}
static void encodeLinear(id<MTLCommandBuffer> command,id<MTLBuffer> input,
    MPSMatrix *weight,id<MTLBuffer> output,size_t rows) {
    [kernel(rows,weight.rows,weight.columns,YES,0) encodeToCommandBuffer:command
        leftMatrix:view(input,rows,weight.columns,weight.columns) rightMatrix:weight
        resultMatrix:view(output,rows,weight.rows,weight.rows)];
}
static id<MTLBuffer> prefixBuffer(const GpuLayer *layer,size_t visible,size_t width,
    size_t cache,size_t index) {
    NSString *key=[NSString stringWithFormat:@"%zu/%zu/%zu",cache,index,visible];
    id<MTLBuffer> data=prefixes[key];
    if (data) return data;
    size_t count=visible*width,length=MAX(count*4,(size_t)4);
    data=[device newBufferWithLength:length options:hostStorage()];
    if (!data) return nil;
    uint16_t *target=data.contents;
    for (size_t i=0;i<count;++i) {
        uint32_t k,v;memcpy(&k,layer->key+i,4);memcpy(&v,layer->value+i,4);
        target[i]=(uint16_t)(k>>16);target[count+i]=(uint16_t)(v>>16);
    }
    if (data.storageMode==MTLStorageModeManaged) [data didModifyRange:NSMakeRange(0,length)];
    if (length<=prefixBudget-prefixBytes) { prefixes[key]=data;prefixBytes+=length; }
    return data;
}
int yue2_metal_stack(const float *input,float *out,const GpuLayer *layers,size_t count,
    size_t rows,size_t hidden,size_t heads,size_t kv,size_t dim,size_t intermediate,
    size_t prefix,size_t cache,float eps,const float *cosine,const float *sine) {
    initialize();if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        if (!attentionInitialize()||!rms||!rotary||!point||!merge) { [lock unlock];return 0; }
        size_t qwidth=heads*dim,kwidth=kv*dim,keys=prefix+rows;
        size_t tile=MIN(rows,(size_t)256),largestLayer=0;
        for (size_t index=0;index<count;++index) {
            size_t bytes=0;
            for (size_t j=0;j<7;++j) bytes+=layers[index].weights[j].rows*layers[index].weights[j].cols*4;
            largestLayer=MAX(largestLayer,bytes);
        }
        size_t working=rows*(3*hidden+2*qwidth+2*kwidth+2*intermediate)*4
            +keys*kwidth*8+heads*tile*dim*8+heads*tile*((keys+15)&~(size_t)15)*4
            +largestLayer*2+weightBudget+prefixBudget;
        if (working>device.recommendedMaxWorkingSetSize*4/5) { [lock unlock];return 0; }
        // These buffers survive every layer; only the final hidden state returns to the CPU.
        id<MTLBuffer> x=buffer(20,rows*hidden*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> norm=buffer(21,rows*hidden*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> q=buffer(22,rows*qwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> k=buffer(23,rows*kwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> v=buffer(25,rows*kwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> gate=buffer(26,rows*intermediate*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> up=buffer(27,rows*intermediate*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> attended=buffer(28,rows*qwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> projected=buffer(29,rows*hidden*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> pk=buffer(8,kv*keys*dim*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> pv=buffer(9,kv*keys*dim*4,MTLResourceStorageModePrivate);
        MPSMatrix *cos=matrix(cosine,rows,dim/2,dim/2,YES,30);
        MPSMatrix *sin=matrix(sine,rows,dim/2,dim/2,YES,31);
        MPSMatrix *initial=matrix(input,rows,hidden,hidden,YES,0);
        if (!x||!norm||!q||!k||!v||!gate||!up||!attended||!projected||!pk||!pv||!cos||!sin||!initial) {
            [lock unlock];return 0;
        }
        for (size_t index=0;index<count;++index) {
            @autoreleasepool {
                const GpuLayer *layer=layers+index;
                id<MTLCommandBuffer> command=[queue commandBuffer];
                if (!command) { [lock unlock];return 0; }
                MPSMatrix *w[7];NSString *names[7];BOOL admit[7];
                for (size_t j=0;j<7;++j) {
                    const GpuWeight *spec=&layer->weights[j];admit[j]=NO;
                    NSString *name=nil;
                    w[j]=converted(spec->raw,spec->dtype,spec->rounded,spec->mapping,
                        spec->rows,spec->cols,command,&name,&admit[j],32+j);
                    names[j]=name;
                    if (!w[j]) { [lock unlock];return 0; }
                }
                id<MTLBuffer> gamma=buffer(24,(hidden*2+dim*2)*4,hostStorage());
                id<MTLBuffer> base=prefixBuffer(layer,prefix,kwidth,cache,index);
                if (!gamma||!base) { [lock unlock];return 0; }
                float *g=gamma.contents;
                memcpy(g,layer->norms[0],hidden*4);memcpy(g+hidden,layer->norms[1],dim*4);
                memcpy(g+hidden+dim,layer->norms[2],dim*4);
                memcpy(g+hidden+dim*2,layer->norms[3],hidden*4);
                if (gamma.storageMode==MTLStorageModeManaged)
                    [gamma didModifyRange:NSMakeRange(0,(hidden*2+dim*2)*4)];
                if (index==0) {
                    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
                    if (!blit) { [lock unlock];return 0; }
                    [blit copyFromBuffer:initial.data sourceOffset:0 toBuffer:x destinationOffset:0 size:rows*hidden*4];
                    [blit endEncoding];
                }
                if (!encodeNorm(command,x,gamma,0,norm,rows,hidden,eps)) { [lock unlock];return 0; }
                encodeLinear(command,norm,w[0],q,rows);encodeLinear(command,norm,w[1],k,rows);
                encodeLinear(command,norm,w[2],v,rows);
                if (!encodeRotary(command,q,gamma,hidden*4,cos.data,sin.data,rows,heads,dim,eps)
                    ||!encodeRotary(command,k,gamma,(hidden+dim)*4,cos.data,sin.data,rows,kv,dim,eps)) {
                    [lock unlock];return 0;
                }
                id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                if (!encoder) { [lock unlock];return 0; }
                MergeShape mergeShape={(uint32_t)rows,(uint32_t)prefix,(uint32_t)kv,(uint32_t)dim};
                [encoder setBuffer:base offset:0 atIndex:0];[encoder setBuffer:k offset:0 atIndex:1];
                [encoder setBuffer:v offset:0 atIndex:2];[encoder setBuffer:pk offset:0 atIndex:3];
                [encoder setBuffer:pv offset:0 atIndex:4];[encoder setBytes:&mergeShape length:sizeof(mergeShape) atIndex:5];
                dispatch(encoder,merge,kv*keys*dim);[encoder endEncoding];
                AttentionShape shape={(uint32_t)rows,(uint32_t)keys,(uint32_t)heads,(uint32_t)kv,
                    (uint32_t)dim,(uint32_t)(heads/kv),0,0,(uint32_t)((keys+15)&~(size_t)15),-1,1.0f/sqrtf(dim)};
                if (!encodeAttention(command,q,pk,pv,attended,shape)) { [lock unlock];return 0; }
                encodeLinear(command,attended,w[3],projected,rows);
                if (!encodePoint(command,x,projected,rows*hidden,2)
                    ||!encodeNorm(command,x,gamma,(hidden+dim*2)*4,norm,rows,hidden,eps)) { [lock unlock];return 0; }
                encodeLinear(command,norm,w[4],gate,rows);encodeLinear(command,norm,w[5],up,rows);
                if (!encodePoint(command,gate,up,rows*intermediate,1)) { [lock unlock];return 0; }
                encodeLinear(command,gate,w[6],projected,rows);
                if (!encodePoint(command,x,projected,rows*hidden,2)) { [lock unlock];return 0; }
                // Bound live weight resources on discrete GPUs instead of retaining all 28 layers in one submission.
                [command commit];[command waitUntilCompleted];
                if (command.status!=MTLCommandBufferStatusCompleted) { [lock unlock];return 0; }
                for (size_t j=0;j<7;++j) {
                    if (admit[j] && w[j].data.length<=weightBudget-weightBytes) {
                        weights[names[j]]=w[j];weightBytes+=w[j].data.length;
                    }
                }
            }
        }
        MPSMatrix *result=matrix(out,rows,hidden,hidden,NO,2);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLBlitCommandEncoder> blit=result&&command ? [command blitCommandEncoder] : nil;
        BOOL ok=blit!=nil;
        if (ok) {
            [blit copyFromBuffer:x sourceOffset:0 toBuffer:result.data destinationOffset:0 size:rows*hidden*4];
            [blit endEncoding];ok=finish(command,result,out,rows,hidden,hidden);
        }
        [lock unlock];return ok;
    }
}

// Raw BF16 weights halve memory traffic for bandwidth-bound token decoding.
int yue2_metal_vector_default(void) {
    initialize();return queue && vectorBudget>=((size_t)2<<30);
}
typedef struct { uint32_t rows,cols,dtype,bf16; } VectorShape;
int yue2_metal_vector(const float *input,const uint8_t *raw,size_t mapping,int dtype,
    int rounded,int priority,float *out,size_t rows,size_t cols) {
    initialize();if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        if (!attentionInitialize()||!weightVector) { [lock unlock];return 0; }
        if (!vectorMode) {
            [weights removeAllObjects];weightBytes=0;vectorMode=YES;
        }
        NSString *name=[NSString stringWithFormat:@"%zu/%p/%zu/%zu/%d",mapping,raw,rows,cols,dtype];
        size_t length=rows*cols*(dtype==1 ? 4 : 2);
        id<MTLBuffer> weight=vectors[name];BOOL admit=NO;
        id<MTLCommandBuffer> command=[queue commandBuffer];
        MPSMatrix *x=matrix(input,1,cols,cols,YES,48);
        MPSMatrix *result=matrix(out,1,rows,rows,NO,49);
        if (!command||!x||!result) { [lock unlock];return 0; }
        if (!weight) admit=length<=vectorBudget-vectorBytes
            && (priority!=2 || vectorBytes>=vectorBudget/2);
        // A large uncached vocabulary streams bounded panels, avoiding two
        // complete GPU copies alongside the protected transformer weights.
        size_t panel=(!weight&&!admit) ? MAX((size_t)1,((size_t)128<<20)/(cols*(dtype==1 ? 4 : 2))) : rows;
        BOOL ok=YES;
        for (size_t start=0;start<rows;start+=panel) {
            size_t tile=MIN(panel,rows-start),bytes=tile*cols*(dtype==1 ? 4 : 2);
            id<MTLBuffer> active=weight;
            if (!active) {
                id<MTLBuffer> staging=buffer(50,bytes,hostStorage());
                active=admit ? [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate]
                    : buffer(51,bytes,MTLResourceStorageModePrivate);
                if (!staging||!active) { [lock unlock];return 0; }
                memcpy(staging.contents,raw+start*cols*(dtype==1 ? 4 : 2),bytes);
                if (staging.storageMode==MTLStorageModeManaged) [staging didModifyRange:NSMakeRange(0,bytes)];
                id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
                if (!blit) { [lock unlock];return 0; }
                [blit copyFromBuffer:staging sourceOffset:0 toBuffer:active destinationOffset:0 size:bytes];
                [blit endEncoding];
                if (admit) weight=active;
            }
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if (!encoder) { [lock unlock];return 0; }
            VectorShape shape={(uint32_t)tile,(uint32_t)cols,(uint32_t)dtype,(uint32_t)rounded};
            [encoder setBuffer:active offset:0 atIndex:0];[encoder setBuffer:x.data offset:0 atIndex:1];
            [encoder setBuffer:result.data offset:start*4 atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
            [encoder setComputePipelineState:weightVector];
            [encoder dispatchThreadgroups:MTLSizeMake(tile,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];
            if (start+tile==rows) ok=finish(command,result,out,1,rows,rows);
            else {
                [command commit];[command waitUntilCompleted];
                ok=command.status==MTLCommandBufferStatusCompleted;
                command=[queue commandBuffer];
                ok=ok&&command!=nil;
            }
            if (!ok) break;
        }
        if (ok&&admit) { vectors[name]=weight;vectorBytes+=weight.length; }
        [lock unlock];return ok;
    }
}

typedef struct { uint32_t keys,capacity,heads,kv,dim,groups;float scale; } CachedAttentionShape;
static Yue2Context *contextEntry(const float *k,const float *v,size_t keys,
    size_t kv,size_t dim,size_t cache,size_t layer,size_t extra) {
        NSString *name=[NSString stringWithFormat:@"%zu/%zu",cache,layer];
        Yue2Context *entry=arContexts[name];
        if (!entry || entry.capacity<keys+extra || entry.length>keys) {
            size_t capacity=(keys+extra+127)&~(size_t)127,length=capacity*kv*dim*4;
            size_t previous=entry ? entry.data.length : 0;
            if (length>prefixBudget-arContextBytes+previous) {
                if (entry) { [arContexts removeObjectForKey:name];arContextBytes-=previous; }
                return nil;
            }
            id<MTLBuffer> data=[device newBufferWithLength:length options:hostStorage()];
            if (!data) { return nil; }
            entry=[Yue2Context new];entry.data=data;entry.capacity=capacity;entry.length=0;
            arContexts[name]=entry;arContextBytes=arContextBytes-previous+length;
        }
        uint16_t *target=entry.data.contents;
        size_t begin=entry.length,width=kv*dim;
        for (size_t head=0;head<kv;++head) {
            size_t offset=head*entry.capacity*dim;
            for (size_t row=begin;row<keys;++row) {
                for (size_t d=0;d<dim;++d) {
                    uint32_t kb,vb;size_t source=row*width+head*dim+d;
                    memcpy(&kb,k+source,4);memcpy(&vb,v+source,4);
                    target[offset+row*dim+d]=(uint16_t)(kb>>16);
                    target[kv*entry.capacity*dim+offset+row*dim+d]=(uint16_t)(vb>>16);
                }
            }
            if (entry.data.storageMode==MTLStorageModeManaged && keys>begin) {
                [entry.data didModifyRange:NSMakeRange((offset+begin*dim)*2,(keys-begin)*dim*2)];
                [entry.data didModifyRange:NSMakeRange((kv*entry.capacity*dim+offset+begin*dim)*2,(keys-begin)*dim*2)];
            }
        }
        entry.length=keys;
    return entry;
}
int yue2_metal_cached_attention(const float *q,const float *k,const float *v,float *out,
    size_t keys,size_t heads,size_t kv,size_t dim,size_t cache,size_t layer) {
    initialize();if (!queue||keys==0||keys>4096||dim==0||dim>256||kv==0||heads==0||heads%kv!=0) return 0;
    @autoreleasepool {
        [lock lock];
        if (!attentionInitialize()||!cachedAttention) { [lock unlock];return 0; }
        Yue2Context *entry=contextEntry(k,v,keys,kv,dim,cache,layer,0);
        if (!entry) { [lock unlock];return 0; }
        MPSMatrix *query=matrix(q,1,heads*dim,heads*dim,YES,4);
        MPSMatrix *result=matrix(out,1,heads*dim,heads*dim,NO,12);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=query&&result&&command ? [command computeCommandEncoder] : nil;
        if (!encoder) { [lock unlock];return 0; }
        CachedAttentionShape shape={(uint32_t)keys,(uint32_t)entry.capacity,(uint32_t)heads,
            (uint32_t)kv,(uint32_t)dim,(uint32_t)(heads/kv),1.0f/sqrtf(dim)};
        [encoder setBuffer:query.data offset:0 atIndex:0];[encoder setBuffer:entry.data offset:0 atIndex:1];
        [encoder setBuffer:result.data offset:0 atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
        [encoder setComputePipelineState:cachedAttention];
        [encoder dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];BOOL ok=finish(command,result,out,1,heads*dim,heads*dim);
        [lock unlock];return ok;
    }
}

static id<MTLBuffer> rawVectorWeight(const GpuWeight *w,id<MTLCommandBuffer> command,
    size_t slot,NSString **nameOut,BOOL *admit) {
    NSString *name=[NSString stringWithFormat:@"%zu/%p/%zu/%zu/%d",w->mapping,w->raw,w->rows,w->cols,w->dtype];
    *nameOut=name;*admit=NO;
    id<MTLBuffer> data=vectors[name];
    if (data) return data;
    size_t length=w->rows*w->cols*(w->dtype==1 ? 4 : 2);
    *admit=length<=vectorBudget-vectorBytes;
    data=[device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    id<MTLBuffer> staging=buffer(slot,length,hostStorage());
    if (!data||!staging) return nil;
    memcpy(staging.contents,w->raw,length);
    if (staging.storageMode==MTLStorageModeManaged) [staging didModifyRange:NSMakeRange(0,length)];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    if (!blit) return nil;
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:data destinationOffset:0 size:length];
    [blit endEncoding];return data;
}
static BOOL encodeVector(id<MTLCommandBuffer> command,id<MTLBuffer> input,
    id<MTLBuffer> weight,const GpuWeight *w,id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) return NO;
    VectorShape shape={(uint32_t)w->rows,(uint32_t)w->cols,(uint32_t)w->dtype,(uint32_t)w->rounded};
    [encoder setBuffer:weight offset:0 atIndex:0];[encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder setComputePipelineState:weightVector];
    [encoder dispatchThreadgroups:MTLSizeMake(w->rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding];return YES;
}
int yue2_metal_ar_stack(const float *input,float *out,float *shadow,const GpuLayer *layers,
    size_t count,size_t hidden,size_t heads,size_t kv,size_t dim,size_t intermediate,
    size_t prefix,size_t cache,float eps,const float *cosine,const float *sine) {
    initialize();if (!queue||prefix+1>4096) return 0;
    @autoreleasepool {
        [lock lock];
        if (!attentionInitialize()||!rms||!rotary||!point||!cachedAttention||!appendContext||!weightVector) {
            [lock unlock];return 0;
        }
        size_t capacity=(prefix+128)&~(size_t)127,existing=0;
        for (size_t index=0;index<count;++index) {
            Yue2Context *entry=arContexts[[NSString stringWithFormat:@"%zu/%zu",cache,index]];
            existing+=entry ? entry.data.length : 0;
        }
        if (capacity*kv*dim*4*count>prefixBudget-arContextBytes+existing) { [lock unlock];return 0; }
        if (!vectorMode) { [weights removeAllObjects];weightBytes=0;vectorMode=YES; }
        size_t qwidth=heads*dim,kwidth=kv*dim;
        id<MTLBuffer> x=buffer(20,hidden*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> norm=buffer(21,hidden*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> q=buffer(22,qwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> k=buffer(23,kwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> v=buffer(25,kwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> gate=buffer(26,intermediate*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> up=buffer(27,intermediate*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> attended=buffer(28,qwidth*4,MTLResourceStorageModePrivate);
        id<MTLBuffer> projected=buffer(29,hidden*4,MTLResourceStorageModePrivate);
        MPSMatrix *cos=matrix(cosine,1,dim/2,dim/2,YES,30);
        MPSMatrix *sin=matrix(sine,1,dim/2,dim/2,YES,31);
        MPSMatrix *initial=matrix(input,1,hidden,hidden,YES,0);
        MPSMatrix *saved=matrix(shadow,1,count*2*kwidth,count*2*kwidth,NO,19);
        if (!x||!norm||!q||!k||!v||!gate||!up||!attended||!projected||!cos||!sin||!initial||!saved) {
            [lock unlock];return 0;
        }
        BOOL cachedLayers[count];
        for (size_t index=0;index<count;++index) {
            cachedLayers[index]=YES;
            for (size_t j=0;j<7;++j) {
                const GpuWeight *w=&layers[index].weights[j];
                NSString *name=[NSString stringWithFormat:@"%zu/%p/%zu/%zu/%d",w->mapping,w->raw,w->rows,w->cols,w->dtype];
                if (!vectors[name]) cachedLayers[index]=NO;
            }
        }
        // Cached layer runs need no weight staging, so submit them together.
        // Separate norm slices keep CPU uploads from overwriting queued work.
        size_t gammaWidth=hidden*2+dim*2;
        id<MTLBuffer> gamma=buffer(24,count*gammaWidth*4,hostStorage());
        if (!gamma) { [lock unlock];return 0; }
        float *allGamma=gamma.contents;
        for (size_t index=0;index<count;++index) {
            const GpuLayer *layer=layers+index;float *g=allGamma+index*gammaWidth;
            memcpy(g,layer->norms[0],hidden*4);memcpy(g+hidden,layer->norms[1],dim*4);
            memcpy(g+hidden+dim,layer->norms[2],dim*4);memcpy(g+hidden+dim*2,layer->norms[3],hidden*4);
        }
        if (gamma.storageMode==MTLStorageModeManaged) [gamma didModifyRange:NSMakeRange(0,count*gammaWidth*4)];
        BOOL ok=YES;
        id<MTLCommandBuffer> command=nil;
        for (size_t index=0;index<count&&ok;++index) {
            @autoreleasepool {
                const GpuLayer *layer=layers+index;
                Yue2Context *entry=contextEntry(layer->key,layer->value,prefix,kv,dim,cache,index,1);
                if (!command) command=[queue commandBuffer];
                if (!entry||!command) { ok=NO;break; }
                id<MTLBuffer> w[7];NSString *names[7];BOOL admit[7];
                for (size_t j=0;j<7;++j) {
                    NSString *name=nil;
                    w[j]=rawVectorWeight(&layer->weights[j],command,32+j,&name,&admit[j]);names[j]=name;
                    if (!w[j]) { ok=NO;break; }
                }
                if (!ok) break;
                size_t gammaOffset=index*gammaWidth*4;
                if (index==0) {
                    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
                    if (!blit) { ok=NO;break; }
                    [blit copyFromBuffer:initial.data sourceOffset:0 toBuffer:x destinationOffset:0 size:hidden*4];
                    [blit endEncoding];
                }
                ok=encodeNorm(command,x,gamma,gammaOffset,norm,1,hidden,eps)
                    &&encodeVector(command,norm,w[0],&layer->weights[0],q)
                    &&encodeVector(command,norm,w[1],&layer->weights[1],k)
                    &&encodeVector(command,norm,w[2],&layer->weights[2],v)
                    &&encodeRotary(command,q,gamma,gammaOffset+hidden*4,cos.data,sin.data,1,heads,dim,eps)
                    &&encodeRotary(command,k,gamma,gammaOffset+(hidden+dim)*4,cos.data,sin.data,1,kv,dim,eps);
                if (!ok) break;
                id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                if (!encoder) { ok=NO;break; }
                struct { uint32_t position,capacity,kv,dim,layer; } append={
                    (uint32_t)prefix,(uint32_t)entry.capacity,(uint32_t)kv,(uint32_t)dim,(uint32_t)index};
                [encoder setBuffer:k offset:0 atIndex:0];[encoder setBuffer:v offset:0 atIndex:1];
                [encoder setBuffer:entry.data offset:0 atIndex:2];[encoder setBuffer:saved.data offset:0 atIndex:3];
                [encoder setBytes:&append length:sizeof(append) atIndex:4];
                dispatch(encoder,appendContext,kwidth);[encoder endEncoding];
                encoder=[command computeCommandEncoder];
                if (!encoder) { ok=NO;break; }
                CachedAttentionShape shape={(uint32_t)(prefix+1),(uint32_t)entry.capacity,(uint32_t)heads,
                    (uint32_t)kv,(uint32_t)dim,(uint32_t)(heads/kv),1.0f/sqrtf(dim)};
                [encoder setBuffer:q offset:0 atIndex:0];[encoder setBuffer:entry.data offset:0 atIndex:1];
                [encoder setBuffer:attended offset:0 atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
                [encoder setComputePipelineState:cachedAttention];
                [encoder dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [encoder endEncoding];
                ok=encodeVector(command,attended,w[3],&layer->weights[3],projected)
                    &&encodePoint(command,x,projected,hidden,2)
                    &&encodeNorm(command,x,gamma,gammaOffset+(hidden+dim*2)*4,norm,1,hidden,eps)
                    &&encodeVector(command,norm,w[4],&layer->weights[4],gate)
                    &&encodeVector(command,norm,w[5],&layer->weights[5],up)
                    &&encodePoint(command,gate,up,intermediate,1)
                    &&encodeVector(command,gate,w[6],&layer->weights[6],projected)
                    &&encodePoint(command,x,projected,hidden,2);
                if (!ok) break;
                entry.length=prefix+1;
                if (!cachedLayers[index] || index+1==count || !cachedLayers[index+1]) {
                    [command commit];[command waitUntilCompleted];ok=command.status==MTLCommandBufferStatusCompleted;
                    command=nil;
                    if (!ok) break;
                    for (size_t j=0;j<7;++j) {
                        if (admit[j]&&w[j].length<=vectorBudget-vectorBytes) {vectors[names[j]]=w[j];vectorBytes+=w[j].length;}
                    }
                }
            }
        }
        if (ok) {
            MPSMatrix *result=matrix(out,1,hidden,hidden,NO,2);
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLBlitCommandEncoder> blit=result&&command ? [command blitCommandEncoder] : nil;
            ok=blit!=nil;
            if (ok) {
                [blit copyFromBuffer:x sourceOffset:0 toBuffer:result.data destinationOffset:0 size:hidden*4];
                if (saved.data.storageMode==MTLStorageModeManaged) [blit synchronizeResource:saved.data];
                [blit endEncoding];ok=finish(command,result,out,1,hidden,hidden);
                if (ok) {
                    memcpy(shadow,saved.data.contents,count*2*kwidth*4);
                    // Managed CPU and GPU copies must agree before a later CPU
                    // append touches a neighbouring part of the same page.
                    for (size_t index=0;index<count;++index) {
                        Yue2Context *entry=arContexts[[NSString stringWithFormat:@"%zu/%zu",cache,index]];
                        uint16_t *target=entry.data.contents;
                        for (size_t head=0;head<kv;++head) {
                            size_t offset=head*entry.capacity*dim+prefix*dim;
                            for (size_t d=0;d<dim;++d) {
                                uint32_t kb,vb;size_t source=index*2*kwidth+head*dim+d;
                                memcpy(&kb,shadow+source,4);memcpy(&vb,shadow+source+kwidth,4);
                                target[offset+d]=(uint16_t)(kb>>16);
                                target[kv*entry.capacity*dim+offset+d]=(uint16_t)(vb>>16);
                            }
                            if (entry.data.storageMode==MTLStorageModeManaged) {
                                [entry.data didModifyRange:NSMakeRange(offset*2,dim*2)];
                                [entry.data didModifyRange:NSMakeRange((kv*entry.capacity*dim+offset)*2,dim*2)];
                            }
                        }
                    }
                }
            }
        }
        // CPU fallback will recompute this token, so discard any partially
        // appended GPU context before it observes a different rounding path.
        if (!ok) { [arContexts removeAllObjects];arContextBytes=0; }
        [lock unlock];return ok;
    }
}
