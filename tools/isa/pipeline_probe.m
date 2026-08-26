// Build a compute pipeline from a .metallib, report the occupancy-relevant
// numbers the driver decides, and serialize a MTLBinaryArchive holding the
// native AGX code.
//
//   clang pipeline_probe.m -O2 -framework Metal -framework Foundation \
//         -fobjc-arc -o pipeline_probe
//   ./pipeline_probe --archive out.bin --json out.json kernel.metallib
//
// Requires a Metal device, so this is a GPU-queue step (docs/ISA-QUEUE.md).
// maxTotalThreadsPerThreadgroup is the driver's own statement of how many
// threads fit given the register footprint of the compiled function, which is
// the closest thing to a register count reachable without disassembly.

#import <Metal/Metal.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#if !__has_feature(objc_arc)
#error compile with -fobjc-arc
#endif

static void die(NSError *err, const char *what) {
  if (err) {
    fprintf(stderr, "%s: %s\n", what, [[err localizedDescription] UTF8String]);
    exit(EXIT_FAILURE);
  }
}

static const struct option longOpts[] = {
    {"archive", required_argument, NULL, 'a'},
    {"json", required_argument, NULL, 'j'},
    {"function", required_argument, NULL, 'f'},
    {NULL, 0, NULL, 0},
};

int main(int argc, char *argv[]) {
  const char *archivePath = NULL;
  const char *jsonPath = NULL;
  const char *onlyFunction = NULL;
  int c;
  while ((c = getopt_long(argc, argv, "a:j:f:", longOpts, NULL)) > 0) {
    switch (c) {
    case 'a': archivePath = optarg; break;
    case 'j': jsonPath = optarg; break;
    case 'f': onlyFunction = optarg; break;
    default:
      fprintf(stderr, "usage: %s [--archive out.bin] [--json out.json] "
                      "[--function name] input.metallib\n", argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (optind >= argc) {
    fprintf(stderr, "need a .metallib\n");
    return EXIT_FAILURE;
  }

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) {
    fprintf(stderr, "no Metal device\n");
    return EXIT_FAILURE;
  }

  NSError *err = nil;
  NSString *path = [NSString stringWithUTF8String:argv[optind]];
  NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&err];
  die(err, "read metallib");
  dispatch_data_t dd = dispatch_data_create([data bytes], [data length],
                                            dispatch_get_main_queue(), ^{});
  id<MTLLibrary> lib = [dev newLibraryWithData:dd error:&err];
  die(err, "load metallib");

  id<MTLBinaryArchive> arc = nil;
  if (archivePath) {
    arc = [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                        error:&err];
    die(err, "make binary archive");
  }

  NSMutableArray *rows = [NSMutableArray array];
  for (NSString *name in [lib functionNames]) {
    if (onlyFunction && strcmp(onlyFunction, [name UTF8String]) != 0) continue;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn || [fn functionType] != MTLFunctionTypeKernel) continue;

    MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
    [desc setComputeFunction:fn];
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithDescriptor:desc
                                           options:MTLPipelineOptionNone
                                        reflection:NULL
                                             error:&err];
    die(err, "make pipeline");

    [rows addObject:@{
      @"function" : name,
      @"maxTotalThreadsPerThreadgroup" : @([ps maxTotalThreadsPerThreadgroup]),
      @"threadExecutionWidth" : @([ps threadExecutionWidth]),
      @"staticThreadgroupMemoryLength" : @([ps staticThreadgroupMemoryLength]),
      @"device" : [dev name],
    }];
    printf("%-52s maxTPTG=%4lu execWidth=%2lu tgMem=%5lu\n", [name UTF8String],
           (unsigned long)[ps maxTotalThreadsPerThreadgroup],
           (unsigned long)[ps threadExecutionWidth],
           (unsigned long)[ps staticThreadgroupMemoryLength]);

    if (arc) {
      [arc addComputePipelineFunctionsWithDescriptor:desc error:&err];
      die(err, "add to archive");
    }
  }

  if ([rows count] == 0) {
    fprintf(stderr, "no kernel functions found in %s\n", argv[optind]);
    return EXIT_FAILURE;
  }

  if (arc) {
    [arc serializeToURL:[NSURL fileURLWithPath:
                                  [NSString stringWithUTF8String:archivePath]]
                  error:&err];
    die(err, "serialize archive");
  }
  if (jsonPath) {
    NSData *out = [NSJSONSerialization dataWithJSONObject:rows
                                                  options:NSJSONWritingPrettyPrinted
                                                    error:&err];
    die(err, "encode json");
    [out writeToFile:[NSString stringWithUTF8String:jsonPath] atomically:YES];
  }
  return 0;
}
