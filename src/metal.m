// Thin Objective-C bridge to Apple's matrix kernels. Model math lives in Rust.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string.h>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static NSLock *lock;
static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *kernels;

static void initialize(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
        if (device && MPSSupportsMTLDevice(device)) {
            queue = [device newCommandQueue];
            lock = [NSLock new];
            kernels = [NSMutableDictionary new];
        } else { device = nil; }
    });
}
int yue2_metal_available(void) { initialize(); return queue != nil; }
const char *yue2_metal_name(void) { initialize(); return device ? device.name.UTF8String : "CPU"; }

// Each buffer is bounded by the caller's matrix tile. Discrete GPUs use a
// managed CPU/VRAM pair; unified GPUs share memory. Completion guards reads.
static MPSMatrix *matrix(const float *data, size_t rows, size_t cols,
                         size_t stride, BOOL copy) {
    size_t packed = (cols + 15) & ~(size_t)15;
    MTLResourceOptions storage = device.hasUnifiedMemory ? MTLResourceStorageModeShared
                                                        : MTLResourceStorageModeManaged;
    id<MTLBuffer> buffer = [device newBufferWithLength:rows * packed * sizeof(float)
                                           options:storage];
    if (!buffer) return nil;
    if (copy) {
        float *target = buffer.contents;
        for (size_t r=0; r<rows; ++r)
            memcpy(target + r*packed, data + r*stride, cols*sizeof(float));
        if (buffer.storageMode == MTLStorageModeManaged)
            [buffer didModifyRange:NSMakeRange(0, buffer.length)];
    }
    MPSMatrixDescriptor *descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                columns:cols rowBytes:packed*sizeof(float) dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
}
int yue2_metal_gemm(const float *a, const float *b, float *c,
                   size_t m, size_t n, size_t k, int transpose_b,
                   size_t lda, size_t ldb, size_t ldc, float beta) {
    initialize(); if (!queue) return 0;
    @autoreleasepool {
        [lock lock];
        MPSMatrix *left = matrix(a,m,k,lda,YES);
        MPSMatrix *right = matrix(b,transpose_b?n:k,transpose_b?k:n,ldb,YES);
        MPSMatrix *result = matrix(c,m,n,ldc,beta!=0);
        if (!left || !right || !result) { [lock unlock]; return 0; }
        NSString *key = [NSString stringWithFormat:@"%zu/%zu/%zu/%d/%g",m,n,k,transpose_b,beta];
        MPSMatrixMultiplication *kernel = kernels[key];
        if (!kernel) {
            kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                transposeLeft:NO transposeRight:transpose_b!=0
                resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:beta];
            kernels[key] = kernel;
            // Avoid unbounded descriptors when processing many distinct sizes.
            if (kernels.count > 256) [kernels removeAllObjects];
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command) { [lock unlock]; return 0; }
        [kernel encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:result];
        if (result.data.storageMode == MTLStorageModeManaged) {
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (!blit) { [lock unlock]; return 0; }
            [blit synchronizeResource:result.data];
            [blit endEncoding];
        }
        [command commit]; [command waitUntilCompleted];
        int ok = command.status == MTLCommandBufferStatusCompleted;
        if (ok) {
            const float *source = result.data.contents;
            size_t stride = result.rowBytes / sizeof(float);
            for (size_t r=0; r<m; ++r) memcpy(c+r*ldc, source+r*stride, n*sizeof(float));
        }
        [lock unlock]; return ok;
    }
}
