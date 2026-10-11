#include <metal_stdlib>
using namespace metal;
struct AttentionShape {
    uint rows, keys, heads, kv_heads, dim, groups;
    uint start, tile, score_stride;
    int causal;
    float scale;
};
kernel void pack_queries(device const float *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         constant AttentionShape &p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i>=p.heads*p.tile*p.dim) return;
    uint d=i%p.dim, r=(i/p.dim)%p.tile;
    uint head=i/(p.dim*p.tile);
    output[i]=input[(p.start+r)*p.heads*p.dim+head*p.dim+d];
}
kernel void pack_keys_values(device const float *key [[buffer(0)]],
                             device const float *value [[buffer(1)]],
                             device float *packed_key [[buffer(2)]],
                             device float *packed_value [[buffer(3)]],
                             constant AttentionShape &p [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if (i>=p.kv_heads*p.keys*p.dim) return;
    uint d=i%p.dim, r=(i/p.dim)%p.keys, head=i/(p.dim*p.keys);
    uint source=r*p.kv_heads*p.dim+head*p.dim+d;
    packed_key[i]=key[source]; packed_value[i]=value[source];
}
kernel void attention_softmax(device float *scores [[buffer(0)]],
                              constant AttentionShape &p [[buffer(1)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[256];
    uint query=p.start+(row%(p.groups*p.tile))%p.tile;
    uint visible=p.causal<0 ? p.keys : min(p.keys,uint(p.causal)+query+1);
    device float *data=scores+row*p.score_stride;
    float maximum=-INFINITY;
    for (uint i=lane;i<visible;i+=256) maximum=max(maximum,data[i]*p.scale);
    reduction[lane]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]=max(reduction[lane],reduction[lane+span]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum=reduction[0];
    float total=0;
    for (uint i=lane;i<p.keys;i+=256) {
        float v=i<visible ? exp(data[i]*p.scale-maximum) : 0;
        data[i]=v; total+=v;
    }
    reduction[lane]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    total=reduction[0];
    for (uint i=lane;i<p.keys;i+=256) data[i]/=total;
}
kernel void unpack_attention(device const float *input [[buffer(0)]],
                             device float *output [[buffer(1)]],
                             constant AttentionShape &p [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i>=p.heads*p.tile*p.dim) return;
    uint d=i%p.dim, r=(i/p.dim)%p.tile, head=i/(p.dim*p.tile);
    uint bits=as_type<uint>(input[i]);
    bits=(bits&0x7fffffff)>0x7f800000 ? (bits&0xffff0000)|0x00400000
        : (bits+0x7fff+((bits>>16)&1))&0xffff0000;
    output[(p.start+r)*p.heads*p.dim+head*p.dim+d]=as_type<float>(bits);
}

// Preserve checkpoint BF16 boundaries while intermediates remain on the device.
inline float rounded(float value) {
    uint bits=as_type<uint>(value);
    bits=(bits&0x7fffffff)>0x7f800000 ? (bits&0xffff0000)|0x00400000
        : (bits+0x7fff+((bits>>16)&1))&0xffff0000;
    return as_type<float>(bits);
}
struct NormShape { uint rows, cols; float eps; };
kernel void rms_norm(device const float *input [[buffer(0)]],
                     device const float *gamma [[buffer(1)]],
                     device float *output [[buffer(2)]],
                     constant NormShape &p [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[256];
    float sum=0;
    for (uint d=lane;d<p.cols;d+=256) { float v=input[row*p.cols+d]; sum+=v*v; }
    reduction[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale=rounded(1.0f/sqrt(reduction[0]/float(p.cols)+p.eps));
    for (uint d=lane;d<p.cols;d+=256)
        output[row*p.cols+d]=rounded(rounded(input[row*p.cols+d]*scale)*gamma[d]);
}
struct HeadShape { uint rows, heads, dim; float eps; };
kernel void normalize_rotary(device float *input [[buffer(0)]],
                             device const float *gamma [[buffer(1)]],
                             device const float *cosine [[buffer(2)]],
                             device const float *sine [[buffer(3)]],
                             constant HeadShape &p [[buffer(4)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[256];
    uint base=head*p.dim, half_dim=p.dim/2, row=head/p.heads;
    float sum=0;
    for (uint d=lane;d<p.dim;d+=256) { float v=rounded(input[base+d]); sum+=v*v; }
    reduction[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale=rounded(1.0f/sqrt(reduction[0]/float(p.dim)+p.eps));
    for (uint d=lane;d<half_dim;d+=256) {
        float a=rounded(rounded(rounded(input[base+d])*scale)*gamma[d]);
        float b=rounded(rounded(rounded(input[base+half_dim+d])*scale)*gamma[half_dim+d]);
        float c=cosine[row*half_dim+d], s=sine[row*half_dim+d];
        input[base+d]=rounded(rounded(a*c)-rounded(b*s));
        input[base+half_dim+d]=rounded(rounded(b*c)+rounded(a*s));
    }
}
struct PointShape { uint count, mode; };
kernel void pointwise(device float *a [[buffer(0)]],
                      device const float *b [[buffer(1)]],
                      constant PointShape &p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i>=p.count) return;
    if (p.mode==0) a[i]=rounded(a[i]);
    else if (p.mode==1) {
        float gate=rounded(a[i]);
        float activated=rounded(gate/(1.0f+exp(-gate)));
        a[i]=rounded(activated*rounded(b[i]));
    } else a[i]=rounded(a[i]+rounded(b[i]));
}
struct MergeShape { uint rows, prefix, kv, dim; };
kernel void merge_keys_values(device const ushort *prefix [[buffer(0)]],
                              device const float *key [[buffer(1)]],
                              device const float *value [[buffer(2)]],
                              device float *packed_key [[buffer(3)]],
                              device float *packed_value [[buffer(4)]],
                              constant MergeShape &p [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    uint keys=p.prefix+p.rows;
    if (i>=p.kv*keys*p.dim) return;
    uint d=i%p.dim, r=(i/p.dim)%keys, head=i/(p.dim*keys);
    if (r<p.prefix) {
        uint source=r*p.kv*p.dim+head*p.dim+d;
        packed_key[i]=as_type<float>(uint(prefix[source])<<16);
        packed_value[i]=as_type<float>(uint(prefix[p.prefix*p.kv*p.dim+source])<<16);
    } else {
        uint source=(r-p.prefix)*p.kv*p.dim+head*p.dim+d;
        packed_key[i]=key[source]; packed_value[i]=rounded(value[source]);
    }
}

struct WeightShape { uint rows, cols, stride, dtype, bf16; };
kernel void convert_weight(device const uchar *raw [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant WeightShape &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i>=p.rows*p.stride) return;
    uint col=i%p.stride, row=i/p.stride;
    float value=0;
    if (col<p.cols) {
        uint source=row*p.cols+col;
        if (p.dtype==0) value=as_type<float>(uint(((device const ushort *)raw)[source])<<16);
        else if (p.dtype==1) value=((device const float *)raw)[source];
        else value=float(((device const half *)raw)[source]);
        if (p.bf16 && p.dtype!=0) value=rounded(value);
    }
    output[i]=value;
}

struct VectorShape { uint rows, cols, dtype, bf16; };
kernel void weight_vector(device const uchar *weight [[buffer(0)]],
                          device const float *input [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant VectorShape &p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float reduction[128];
    float sum=0;
    for (uint col=lane;col<p.cols;col+=128) {
        uint index=row*p.cols+col;
        float v;
        if (p.dtype==0) v=as_type<float>(uint(((device const ushort *)weight)[index])<<16);
        else if (p.dtype==1) v=((device const float *)weight)[index];
        else v=float(((device const half *)weight)[index]);
        if (p.bf16 && p.dtype!=0) v=rounded(v);
        sum+=v*input[col];
    }
    reduction[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=64;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane==0) output[row]=reduction[0];
}

struct CachedAttentionShape { uint keys, capacity, heads, kv, dim, groups; float scale; };
kernel void cached_attention(device const float *query [[buffer(0)]],
                             device const ushort *context [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant CachedAttentionShape &p [[buffer(3)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scores[4096];
    threadgroup float reduction[256];
    uint base=(head/p.groups)*p.capacity*p.dim;
    float maximum=-INFINITY;
    for (uint row=lane;row<p.keys;row+=256) {
        float sum=0;
        for (uint d=0;d<p.dim;++d)
            sum+=query[head*p.dim+d]*as_type<float>(uint(context[base+row*p.dim+d])<<16);
        scores[row]=sum*p.scale;
        maximum=max(maximum,scores[row]);
    }
    reduction[lane]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]=max(reduction[lane],reduction[lane+span]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum=reduction[0];
    float total=0;
    for (uint row=lane;row<p.keys;row+=256) { scores[row]=exp(scores[row]-maximum);total+=scores[row]; }
    reduction[lane]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    total=reduction[0];
    for (uint row=lane;row<p.keys;row+=256) scores[row]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane<p.dim) {
        uint value_base=p.kv*p.capacity*p.dim+base+lane;
        float sum=0;
        for (uint row=0;row<p.keys;++row)
            sum+=scores[row]*as_type<float>(uint(context[value_base+row*p.dim])<<16);
        output[head*p.dim+lane]=rounded(sum);
    }
}

// Split long single-query attention into bounded 1024-key partitions. Each
// partition retains its own softmax maximum, sum and unnormalised value sum;
// a second kernel merges them with a numerically stable global softmax.
kernel void cached_attention_parts(device const float *query [[buffer(0)]],
                                   device const ushort *context [[buffer(1)]],
                                   device float *partials [[buffer(2)]],
                                   constant CachedAttentionShape &p [[buffer(3)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scores[1024];
    threadgroup float reduction[256];
    uint head=group.x, part=group.y, start=part*1024;
    uint rows=min(1024u,p.keys-start), parts=(p.keys+1023)/1024;
    uint base=(head/p.groups)*p.capacity*p.dim;
    float maximum=-INFINITY;
    for (uint row=lane;row<rows;row+=256) {
        float sum=0;
        for (uint d=0;d<p.dim;++d)
            sum+=query[head*p.dim+d]*as_type<float>(uint(context[base+(start+row)*p.dim+d])<<16);
        scores[row]=sum*p.scale;
        maximum=max(maximum,scores[row]);
    }
    reduction[lane]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]=max(reduction[lane],reduction[lane+span]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum=reduction[0];
    float total=0;
    for (uint row=lane;row<rows;row+=256) {
        scores[row]=exp(scores[row]-maximum);
        total+=scores[row];
    }
    reduction[lane]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span=128;span>0;span>>=1) {
        if (lane<span) reduction[lane]+=reduction[lane+span];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output=(head*parts+part)*(p.dim+2);
    if (lane==0) { partials[output]=maximum;partials[output+1]=reduction[0]; }
    if (lane<p.dim) {
        uint value_base=p.kv*p.capacity*p.dim+base+lane;
        float sum=0;
        for (uint row=0;row<rows;++row)
            sum+=scores[row]*as_type<float>(uint(context[value_base+(start+row)*p.dim])<<16);
        partials[output+2+lane]=sum;
    }
}
kernel void cached_attention_merge(device const float *partials [[buffer(0)]],
                                   device float *output [[buffer(1)]],
                                   constant CachedAttentionShape &p [[buffer(2)]],
                                   uint head [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    if (lane>=p.dim) return;
    uint parts=(p.keys+1023)/1024, stride=p.dim+2, base=head*parts*stride;
    float maximum=-INFINITY;
    for (uint part=0;part<parts;++part) maximum=max(maximum,partials[base+part*stride]);
    float total=0,sum=0;
    for (uint part=0;part<parts;++part) {
        uint at=base+part*stride;
        float weight=exp(partials[at]-maximum);
        total+=weight*partials[at+1];
        sum+=weight*partials[at+2+lane];
    }
    output[head*p.dim+lane]=rounded(sum/total);
}

struct AppendShape { uint position, capacity, kv, dim, layer; };
kernel void append_context(device const float *key [[buffer(0)]],
                           device const float *value [[buffer(1)]],
                           device ushort *context [[buffer(2)]],
                           device float *shadow [[buffer(3)]],
                           constant AppendShape &p [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    if (i>=p.kv*p.dim) return;
    uint head=i/p.dim,d=i%p.dim,base=head*p.capacity*p.dim+p.position*p.dim+d;
    float k=key[i],v=rounded(value[i]);
    context[base]=ushort(as_type<uint>(k)>>16);
    context[p.kv*p.capacity*p.dim+base]=ushort(as_type<uint>(v)>>16);
    shadow[p.layer*2*p.kv*p.dim+i]=k;
    shadow[(p.layer*2+1)*p.kv*p.dim+i]=v;
}
