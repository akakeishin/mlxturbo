// fastmlx bridge — B1 probe の下部工事。
//
// MLX が確保した MTLBuffer を受け取り、fastmlx 自身の MTLCommandQueue 上で
// 1 本の MTLCommandBuffer に N 個の compute dispatch を直接エンコードして
// submit する最小経路。C ABI のみを公開し、Python からは ctypes で叩く
// (pybind11 / nanobind に依存しない = pyproject を触らずに済む)。
//
// MTLBuffer の取得は Python 側 (bridge.py) が mx.array.__dlpack__() から行う。
// MLX の DLPack は device_type=kDLMetal(8) で、DLTensor.data に MTLBuffer の
// ポインタ、byte_offset にバイト単位のオフセットを入れてくる。ここではその
// ポインタを不透明な void* として受け取るだけで、libmlx へのリンクは不要。
//
// ビルド: ./build.sh

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define FMB_EXPORT extern "C" __attribute__((visibility("default")))

// ---------------------------------------------------------------------------
// submit のフラグ
// ---------------------------------------------------------------------------
enum {
  FMB_WAIT = 1 << 0,          // waitUntilCompleted まで行う
  FMB_THREADGROUPS = 1 << 1,  // grid を threadgroup 数として dispatchThreadgroups
  FMB_NO_BARRIER = 1 << 2,    // dispatch 間の memoryBarrier を省く
  FMB_SPLIT_CB = 1 << 3,      // dispatch ごとに command buffer を分ける
  FMB_SPLIT_ENCODER = 1 << 4, // dispatch ごとに encoder を分ける (CB は 1 本)
  FMB_UNRETAINED = 1 << 5,    // commandBufferWithUnretainedReferences を使う
  FMB_ORDER_CB = 1 << 6,      // 分割した command buffer 間を MTLEvent で直列化
};

// ---------------------------------------------------------------------------
// 1 dispatch の記述。Python 側 (ctypes.Structure) と 1 バイトずれず一致させること。
// ---------------------------------------------------------------------------
typedef struct {
  int32_t pipeline;             //  0: fmb_pipeline が返した id
  int32_t n_buffers;            //  4: バインドする MTLBuffer の本数
  const void* const* buffers;   //  8: MTLBuffer* の配列 (長さ n_buffers)
  const uint64_t* offsets;      // 16: 各 buffer のバイトオフセット (NULL 可 = 全部 0)
  const void* bytes;            // 24: setBytes で渡す小さな定数ブロブ (NULL 可)
  uint32_t bytes_len;           // 32
  int32_t bytes_index;          // 36: bytes のバインド index (-1 で無効)
  uint32_t grid_x, grid_y, grid_z;  // 40,44,48
  uint32_t tg_x, tg_y, tg_z;        // 52,56,60
  uint32_t threadgroup_mem_len;     // 64
  int32_t threadgroup_mem_index;    // 68: (-1 で無効)
} FMBDispatch;                      // = 72 bytes

struct FMBContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLEvent> order_event;  // FMB_ORDER_CB 用。遅延生成
  uint64_t event_value = 0;
  std::vector<id<MTLLibrary>> libs;
  std::vector<id<MTLComputePipelineState>> pipes;
  std::string device_name;
  std::string last_error;
  // 直前の submit の計測値
  double last_encode_ms = 0.0;  // encode 開始 ~ 全 commit 完了
  double last_wall_ms = 0.0;    // encode 開始 ~ (wait 指定時は) 完了待ちまで
  double last_gpu_ms = 0.0;     // GPUEndTime - GPUStartTime の総和
  int last_command_buffers = 0;
};

namespace {

void set_err(char* err, int errlen, NSString* msg) {
  if (!err || errlen <= 0) {
    return;
  }
  const char* c = msg ? [msg UTF8String] : "unknown error";
  if (!c) {
    c = "unknown error";
  }
  std::strncpy(err, c, (size_t)errlen - 1);
  err[errlen - 1] = '\0';
}

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
      .count();
}

}  // namespace

// ---------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------

// buffer_hint に MLX 由来の MTLBuffer を渡すと、その buffer を所有している
// MTLDevice からキューを作る。Metal はリソースとキューの device 一致を要求
// するので、MTLCreateSystemDefaultDevice() の戻りを当てにせずこちらを使う。
FMB_EXPORT FMBContext* fmb_context_create(void* buffer_hint, char* err, int errlen) {
  @autoreleasepool {
    id<MTLDevice> dev = nil;
    if (buffer_hint) {
      id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer_hint;
      if (![b conformsToProtocol:@protocol(MTLBuffer)]) {
        set_err(err, errlen, @"buffer_hint is not an MTLBuffer");
        return nullptr;
      }
      dev = [b device];
    } else {
      dev = MTLCreateSystemDefaultDevice();
    }
    if (!dev) {
      set_err(err, errlen, @"no MTLDevice");
      return nullptr;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) {
      set_err(err, errlen, @"newCommandQueue failed");
      return nullptr;
    }
    FMBContext* ctx = new FMBContext();
    ctx->device = dev;
    ctx->queue = q;
    ctx->device_name = [[dev name] UTF8String];
    return ctx;
  }
}

FMB_EXPORT void fmb_context_destroy(FMBContext* ctx) {
  if (!ctx) {
    return;
  }
  @autoreleasepool {
    ctx->libs.clear();
    ctx->pipes.clear();
    ctx->queue = nil;
    ctx->device = nil;
  }
  delete ctx;
}

FMB_EXPORT const char* fmb_device_name(FMBContext* ctx) {
  return ctx ? ctx->device_name.c_str() : "";
}

FMB_EXPORT int fmb_dispatch_struct_size() { return (int)sizeof(FMBDispatch); }

// ---------------------------------------------------------------------------
// buffer の素性確認 (Python 側の検算用)
// ---------------------------------------------------------------------------
FMB_EXPORT int fmb_buffer_info(void* buffer, uint64_t* out_length, void** out_contents,
                               char* err, int errlen) {
  @autoreleasepool {
    if (!buffer) {
      set_err(err, errlen, @"null buffer");
      return -1;
    }
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffer;
    if (![b conformsToProtocol:@protocol(MTLBuffer)]) {
      set_err(err, errlen, @"not an MTLBuffer");
      return -1;
    }
    if (out_length) {
      *out_length = (uint64_t)[b length];
    }
    if (out_contents) {
      *out_contents = [b contents];
    }
    return 0;
  }
}

// ---------------------------------------------------------------------------
// MSL ソース -> MTLLibrary -> MTLComputePipelineState
// ---------------------------------------------------------------------------
FMB_EXPORT int fmb_library_from_source(FMBContext* ctx, const char* source, int fast_math,
                                       char* err, int errlen) {
  @autoreleasepool {
    if (!ctx || !source) {
      set_err(err, errlen, @"null ctx/source");
      return -1;
    }
    NSError* e = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    NSString* src = [NSString stringWithUTF8String:source];
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:opts error:&e];
    if (!lib) {
      set_err(err, errlen,
              e ? [e localizedDescription] : @"newLibraryWithSource failed");
      return -1;
    }
    ctx->libs.push_back(lib);
    return (int)ctx->libs.size() - 1;
  }
}

FMB_EXPORT int fmb_pipeline(FMBContext* ctx, int lib_id, const char* fn_name, char* err,
                            int errlen) {
  @autoreleasepool {
    if (!ctx || lib_id < 0 || lib_id >= (int)ctx->libs.size() || !fn_name) {
      set_err(err, errlen, @"bad lib id / fn name");
      return -1;
    }
    NSString* name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [ctx->libs[(size_t)lib_id] newFunctionWithName:name];
    if (!fn) {
      set_err(err, errlen,
              [NSString stringWithFormat:@"no such kernel function: %@", name]);
      return -1;
    }
    NSError* e = nil;
    id<MTLComputePipelineState> ps =
        [ctx->device newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) {
      set_err(err, errlen, e ? [e localizedDescription] : @"pipeline creation failed");
      return -1;
    }
    ctx->pipes.push_back(ps);
    return (int)ctx->pipes.size() - 1;
  }
}

FMB_EXPORT int fmb_pipeline_max_threads(FMBContext* ctx, int pipe_id) {
  if (!ctx || pipe_id < 0 || pipe_id >= (int)ctx->pipes.size()) {
    return -1;
  }
  return (int)[ctx->pipes[(size_t)pipe_id] maxTotalThreadsPerThreadgroup];
}

// ---------------------------------------------------------------------------
// 本体: N dispatch を 1 command buffer にエンコードして submit
// ---------------------------------------------------------------------------
FMB_EXPORT int fmb_submit(FMBContext* ctx, const FMBDispatch* ds, int n, int flags,
                          char* err, int errlen) {
  if (!ctx || !ds || n <= 0) {
    set_err(err, errlen, @"null ctx / empty dispatch list");
    return -1;
  }
  for (int i = 0; i < n; ++i) {
    if (ds[i].pipeline < 0 || ds[i].pipeline >= (int)ctx->pipes.size()) {
      set_err(err, errlen, @"dispatch references unknown pipeline id");
      return -1;
    }
  }

  const bool wait = (flags & FMB_WAIT) != 0;
  const bool use_tg = (flags & FMB_THREADGROUPS) != 0;
  const bool barrier = (flags & FMB_NO_BARRIER) == 0;
  const bool split_cb = (flags & FMB_SPLIT_CB) != 0;
  const bool split_enc = (flags & FMB_SPLIT_ENCODER) != 0;
  const bool unretained = (flags & FMB_UNRETAINED) != 0;
  const bool order_cb = (flags & FMB_ORDER_CB) != 0;

  double t0 = now_ms();
  double t_commit = t0;
  double gpu_ms = 0.0;
  int n_cb = 0;
  int rc = 0;

  @autoreleasepool {
    std::vector<id<MTLCommandBuffer>> cbs;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    int cb_index = -1;

    // 同一キューでも command buffer 同士は重なって走る (実測)。依存のある
    // 連鎖を CB 分割で流すなら MTLEvent で明示的に直列化する必要がある。
    id<MTLEvent> ev = nil;
    uint64_t ev_base = 0;
    if (order_cb) {
      if (!ctx->order_event) {
        ctx->order_event = [ctx->device newEvent];
      }
      ev = ctx->order_event;
      ev_base = ctx->event_value;
    }

    auto new_cb = [&]() {
      cb = unretained ? [ctx->queue commandBufferWithUnretainedReferences]
                      : [ctx->queue commandBuffer];
      cbs.push_back(cb);
      ++n_cb;
      ++cb_index;
      if (ev && cb_index > 0) {
        [cb encodeWaitForEvent:ev value:ev_base + (uint64_t)cb_index];
      }
      enc = nil;
    };

    auto close_cb = [&]() {
      if (enc) {
        [enc endEncoding];
        enc = nil;
      }
      if (ev) {
        [cb encodeSignalEvent:ev value:ev_base + (uint64_t)cb_index + 1];
      }
      [cb commit];
      cb = nil;
    };

    for (int i = 0; i < n; ++i) {
      const FMBDispatch& d = ds[i];

      if (cb == nil || (split_cb && i > 0)) {
        if (cb) {
          close_cb();
        }
        new_cb();
      }
      if (enc == nil || (split_enc && i > 0)) {
        if (enc) {
          [enc endEncoding];
        }
        enc = [cb computeCommandEncoder];  // MTLDispatchTypeSerial
        if (!enc) {
          set_err(err, errlen, @"computeCommandEncoder failed");
          rc = -1;
          break;
        }
      } else if (barrier && i > 0) {
        // 同一 encoder 内の連鎖。serial encoder は tracked resource に対して
        // 自動でハザードを解決するが、heap 由来 (MLX の <=256B 割り当て) など
        // untracked なリソースが混ざる可能性があるので明示的に張る。
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [enc setComputePipelineState:ctx->pipes[(size_t)d.pipeline]];
      for (int b = 0; b < d.n_buffers; ++b) {
        id<MTLBuffer> mb = (__bridge id<MTLBuffer>)(void*)d.buffers[b];
        NSUInteger off = d.offsets ? (NSUInteger)d.offsets[b] : 0;
        [enc setBuffer:mb offset:off atIndex:(NSUInteger)b];
      }
      if (d.bytes && d.bytes_len > 0 && d.bytes_index >= 0) {
        [enc setBytes:d.bytes
                length:(NSUInteger)d.bytes_len
               atIndex:(NSUInteger)d.bytes_index];
      }
      if (d.threadgroup_mem_len > 0 && d.threadgroup_mem_index >= 0) {
        [enc setThreadgroupMemoryLength:(NSUInteger)d.threadgroup_mem_len
                                atIndex:(NSUInteger)d.threadgroup_mem_index];
      }

      MTLSize grid = MTLSizeMake(d.grid_x, d.grid_y, d.grid_z);
      MTLSize tg = MTLSizeMake(d.tg_x, d.tg_y, d.tg_z);
      if (use_tg) {
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      } else {
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      }
    }

    if (rc == 0) {
      if (cb) {
        close_cb();
      }
      if (ev) {
        ctx->event_value = ev_base + (uint64_t)n_cb;
      }
      t_commit = now_ms();

      if (wait) {
        for (id<MTLCommandBuffer> c : cbs) {
          [c waitUntilCompleted];
        }
        for (id<MTLCommandBuffer> c : cbs) {
          if ([c status] == MTLCommandBufferStatusError) {
            set_err(err, errlen,
                    [c error] ? [[c error] localizedDescription]
                              : @"command buffer error");
            rc = -1;
          }
          gpu_ms += ([c GPUEndTime] - [c GPUStartTime]) * 1000.0;
        }
      }
    } else {
      // エンコード途中で失敗。開いている CB も signal だけは打って畳む
      // (打たないと ORDER_CB の待ち値が宙に浮く)。完了まで待ってから返す。
      if (cb) {
        close_cb();
      }
      if (ev) {
        ctx->event_value = ev_base + (uint64_t)n_cb;
      }
      for (id<MTLCommandBuffer> c : cbs) {
        [c waitUntilCompleted];
      }
      t_commit = now_ms();
    }
    cbs.clear();
  }

  ctx->last_encode_ms = t_commit - t0;
  ctx->last_wall_ms = now_ms() - t0;
  ctx->last_gpu_ms = gpu_ms;
  ctx->last_command_buffers = n_cb;
  return rc;
}

FMB_EXPORT double fmb_last_encode_ms(FMBContext* ctx) {
  return ctx ? ctx->last_encode_ms : -1.0;
}
FMB_EXPORT double fmb_last_wall_ms(FMBContext* ctx) {
  return ctx ? ctx->last_wall_ms : -1.0;
}
FMB_EXPORT double fmb_last_gpu_ms(FMBContext* ctx) {
  return ctx ? ctx->last_gpu_ms : -1.0;
}
FMB_EXPORT int fmb_last_command_buffers(FMBContext* ctx) {
  return ctx ? ctx->last_command_buffers : -1;
}
