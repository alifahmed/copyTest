#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <immintrin.h>
#include <omp.h>

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

typedef uint64_t u64;
typedef void (*kernel)(void);
typedef struct {
	kernel func;
	const string name;
} testDesc;

high_resolution_clock::time_point tt;

void tic(){
	tt = high_resolution_clock::now();
}

double toc(){
	duration<double, std::milli> ms_double = high_resolution_clock::now() - tt;
	return ms_double.count();
}

void* __restrict src = nullptr;
void* __restrict dst = nullptr;
u64 len = 0;
u64 iter = 0;
vector<testDesc> testKernels;

void prepArray(){
	int srcVal = rand();
	memset(src, srcVal, len);
	memset(dst, 0, len);
}

void testArray(){
	int ret = memcmp(src, dst, len);
	if(ret){
		cout << "\tFailed" << endl;
	}
	else {
		cout << "\tPassed" << endl;
	}
}

void runTests(){
	for(const testDesc& desc : testKernels){
		cout << desc.name << " test..." << endl;
		prepArray();
		tic();
		for(u64 i = 0; i < iter; i++){
			desc.func();
		}
		double time = toc() / 1000;
		cout << "\ttime: " << time << " sec" << endl;
		cout << "\tbandwidth: " << iter * len * 2 / 1024.0 / 1024.0 / 1024.0 / time << " GB/s" << endl;
		testArray();
	}
}

void registerTest(testDesc desc){
	testKernels.push_back(desc);
}

void memcpyTest(){
	memcpy(dst, src, len);
}

void memmoveTest(){
	memmove(dst, src, len);
}

static inline void *__movsb(void *d, const void *s, size_t n) {
  asm volatile ("rep movsb"
                : "=D" (d),
                  "=S" (s),
                  "=c" (n)
                : "0" (d),
                  "1" (s),
                  "2" (n)
                : "memory");
  return d;
}

void ermsbTest(){
	__movsb(dst, src, len);
}

void avx2Test(){
	const u64 ii = len / 32;
	__m256i* __restrict srcV = (__m256i*)__builtin_assume_aligned(src, 64);
	__m256i* __restrict dstV = (__m256i*)__builtin_assume_aligned(dst, 64);
	//#pragma omp parallel for
	for(u64 i = 0; i < ii; i++){
		_mm256_store_si256(&dstV[i], _mm256_load_si256(&srcV[i]));
	}
}

void avx2NTTest(){
	__m256i val;
	const u64 ii = len / 32;
	__m256i* __restrict srcV = (__m256i*)__builtin_assume_aligned(src, 64);
	__m256i* __restrict dstV = (__m256i*)__builtin_assume_aligned(dst, 64);
	//#pragma omp parallel for
	for(u64 i = 0; i < ii; i++){
		_mm256_store_si256(&dstV[i], _mm256_stream_load_si256(&srcV[i]));
	}
}

int main(int argc, const char** argv){
	if(argc != 3){
		cout << "usage: copytest <size_in_GB> <iter>" << endl;
		exit(-1);
	}
	len = atol(argv[1]) << 30;		//convert to GB
	iter = atol(argv[2]);
	
	src = aligned_alloc(64, len);
	dst = aligned_alloc(64, len);
	
	if((src == nullptr) || (dst == nullptr)){
		cout << "malloc failed" << endl;
		exit(-1);
	}

	registerTest({memcpyTest, "memcpy"});
	registerTest({avx2Test, "avx2"});
	registerTest({avx2NTTest, "avx2 non temporal"});
	registerTest({ermsbTest, "ERMSB"});

	runTests();

	free(src);
	free(dst);
	
	return 0;
}
