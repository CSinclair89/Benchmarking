#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <timer.h>
#include <aligned_allocator.h>
#include "partitioner.hpp"
#include "dummy.h"

#ifdef WITH_PAPI
#include <papi.h>
#endif

#ifdef ENABLE_PARALLEL_CXX
#include <parallel/algorithm>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T, class BinaryCompOp>
void selectionSort(const int n, T *arr, const BinaryCompOp &comp) {
   for( int j = 0; j < n - 1 ; ++j)
   {
      int idx = j;
      for (int i = j + 1; i < n; ++i)
         if (comp(arr[i], arr[idx])) idx = i;
      if (j != idx) std::swap(arr[j], arr[idx]);
   }
}

template<typename T, class BinaryCompOp>
void selectionSort(T *first, T *last, const BinaryCompOp &comp) {
   const size_t len = last - first;
   selectionSort(len, first, comp);
}

template<typename ValueType, typename PointerType, class BinaryCompOp>
void selectionSortPtr(const int n, ValueType *arr, PointerType *ptr, const BinaryCompOp &comp) {
   for (int j = 0; j < n - 1; ++j) {
      int idx = j;
      for (int i = j + 1; i < n; ++i) if (comp(arr[ptr[i]], arr[ptr[idx]])) idx=i;
      if (j != idx) std::swap(ptr[j], ptr[idx]);
   }
}

template<typename ValueType, typename PointerType, class BinaryCompOp>
void selectionSortPtr(ValueType *first, ValueType *last, PointerType *ptr, const BinaryCompOp &comp) {
   const size_t len = last - first;
   selectionSortPtr(len, first, ptr, comp);
}

template<typename T, typename P, class BinaryCompOp>
T* myPartition(T *first, T *last, const P &pivot, const BinaryCompOp &comp) {
   while(first != last) {
      while(comp(*first,pivot)) {
         ++first;
         if (first == last) return first;
      }
      do {
         --last;
         if (first == last) return first;
      }
      while (!comp(*last, pivot));
      std::swap(*first, *last);
      ++first;
   }
   return first;
}

#ifdef _OPENMP
template<typename T, typename P, class BinaryCompOp>
T* ParallelPartition(T *data, const size_t len, const P &pivot, const BinaryCompOp &comp) {
   const int maxThreads = omp_get_max_threads();
   if (maxThreads == 1) return myPartition(data, data + len, pivot, comp);
   std::vector<size_t> shared_low_counts(maxThreads, 0);
   auto shared_parts = partition_range(len, maxThreads);
   size_t shared_split_index = 0;
   
   #pragma omp parallel default(shared)
   {
      const int threadID = omp_get_thread_num();
      const size_t my_start = shared_parts[threadID];
      const size_t my_stop = shared_parts[threadID+1];
      const size_t my_size = my_stop - my_start;
      
      std::vector<T> local_data(data + my_start, data + my_stop);
      T* my_data = local_data.data();
      T* my_split = myPartition(my_data, my_data + my_size, pivot, comp);
      
      const size_t my_count = std::distance(my_data, my_split);
      shared_low_counts[threadID] = my_count;
      
      #pragma omp barrier
      size_t global_split_index = 0, my_left_offset = 0, my_right_offset = 0;
      for (int tid = 0; tid < maxThreads; ++tid) {
         global_split_index += shared_low_counts[tid];
         if (tid < threadID) {
            my_left_offset += shared_low_counts[tid];
            const size_t thread_size = shared_parts[tid + 1] - shared_parts[tid];
            my_right_offset += (thread_size - shared_low_counts[tid]);
         }
      }
      
      #pragma omp master
      shared_split_index = global_split_index;
      my_right_offset += global_split_index;
      for (size_t i = 0; i < my_count; ++i) data[i + my_left_offset] = my_data[i];
      const size_t my_right_count = my_size - my_count;
      for (size_t i = 0; i < my_right_count; ++i) data[i + my_right_offset] = my_split[i];
   }
   return data + shared_split_index;
}
#endif

#ifdef ENABLE_SIMD
#include <immintrin.h>

template<typename Comp>
int* myPartitionSimd(int *first, int *last, const int &pivot, const Comp&) {
   std::less<int> comp;
   int *orig_first = first;
   int *orig_last = last;
   const int v_len = sizeof(__m512i) / sizeof(int);
   const __m512i v_pivot = _mm512_set1_epi32(pivot);
   auto all = [](__mmask16& mask) { return uint16_t(mask) == 0xFFFF; };
   auto any = [](__mmask16& mask) { return uint16_t(mask) != 0; };
   while (first != last) {
      while (std::distance(first, last) > v_len) {
         __m512i v = isAligned(first, 64u) ? _mm512_load_si512(first) : _mm512_loadu_si512(first);
         __mmask16 mask = _mm512_cmplt_epi32_mask(v, v_pivot);
         if (all(mask)) first += v_len;
         else break;
      }
      while (comp(*first, pivot)) {
         ++first;
         if (first == last) return first;
      }
      while (std::distance(first, last) > v_len) {
         int *ptr = last - v_len;
         __m512i v = isAligned(ptr, 64u) ? _mm512_load_si512(ptr) : _mm512_loadu_si512(ptr);
         __mmask16 mask = _mm512_cmplt_epi32_mask(v, v_pivot);
         if (any(mask)) break;
         else last -= v_len;
      }
      do {
         --last;
         if (first == last) return first;
      }
      while (!comp(*last, pivot));
      std::swap(*first, *last);
      ++first;
   }
   return first;
}
#endif

template<typename T, class BinaryComparisonOp>
bool myIsSorted(T *first, T *last, const BinaryComparisonOp &comp) {
   auto arr = first;
   auto n = std::distance(first, last);
   int yes = 1;
   for (size_t i = 0; i < n - 1; ++i) yes &= not(comp(arr[i+1], arr[i]));
   return yes;
}

template<typename T>
inline T medianOfThree (const T x, const T y, const T z) {
   if(x < y) {
      if(y < z) return y;
      if(x < z) return z;
   }
   if(y < z) {
      if(z < x) return z;
      if(x < z) return x;
   }
   if(z < x) {
      if(z < y) return y;
      if(x < y) return x;
   }
   return y;
}

template<typename T>
inline T selectPivot(T *first, T *last) {
   if (last - first > 3) {
      const T lo = *first;
      const T mid = first[(last - first) / 2];
      const T hi = *(last - 1);
      return medianOfThree(lo, mid, hi);
   }
   else return first[0];
}

template<typename T>
void printList(T *first, T *last) {
   const int n = last - first;
   printf("[");
   for (int i = 0; i < (n - 1); ++i) std::cout << first[i] << ",";
   if (n > 0) std::cout << first[n - 1];
   printf("]\n");
}

template<typename T, class Comp = std::less<T>>
struct QuickSortImpl {
   int maxParallelDepth = 8;
   size_t selectionSortCutoff = 10;
   const Comp& comp;
   QuickSortImpl (const Comp& comp) : comp(comp) {}
   void run(T *first, T *last, const int level) {
      const auto n = std::distance(first, last);
      if (n <= 1) return;
      else if (n < this->selectionSortCutoff) selectionSort(n, first, this->comp);
      else {
         const T pivotValue = selectPivot(first, last);
         T *middle = myPartition(first, last, pivotValue, comp);
         if (middle == first || middle == last) {
            while (middle < last and !comp(pivotValue, *middle)) middle++;
            if (middle == last) return;
            const T newPivot = *middle;
            middle = myPartition(first, last, newPivot, comp);
         }
         if (middle - first <= last - middle) {
            if (middle != first) {
               #pragma omp task if (level < maxParallelDepth)
               this->run(first, middle, level + 1);
            }
               this->run(middle, last, level + 1);
         }
         else {
            if (middle != last) {
               #pragma omp task if (level < maxParallelDepth)
               this->run(middle, last, level + 1);
            }
               this->run(first, middle, level + 1);
         }
      }
   }
};

template<typename T, class BinaryComp>
void quickSort(T *first, T *last, const BinaryComp& comp) {
   if (first >= last) return;
   QuickSortImpl<T, BinaryComp> op(comp);
   #pragma omp parallel
   {
      #pragma omp single nowait
      op.run(first, last, 
      0);
   }
}

/*===================*/
/*      MY CODE      */
/*===================*/

// Slow Sort
template<typename T, class Comp>
void slowSort(T* first, T* last, const Comp& comp) {
    size_t n = last - first;
    if (n <= 1) return;

    size_t m = (n-1)/2;
    T* mid = first + m;
    
    #pragma omp parallel
    #pragma omp single nowait
    {
      #pragma omp task
      slowSort(first, mid+1, comp);
      
      #pragma omp task
      slowSort(mid+1, last, comp);
      
      #pragma omp taskwait
    }

    if(comp(*(last-1), *mid)) std::swap(*(last-1), *mid);

    slowSort(first, last-1, comp);
}

// Stooge Sort
template<typename T, class Comp>
void stoogeSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  if (n <= 1) return;
  if (comp(*(last - 1), *first)) std::swap(*first, *(last - 1));
  if (n > 2) {
    size_t t = n / 3;
    stoogeSort(first, last - t, comp);
    stoogeSort(first + t, last, comp);
    stoogeSort(first, last - t, comp);    
  }
}

// Pancake Sort
template<typename T, class Comp>
void pancakeSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  if (n <= 1) return;
  
  for (size_t size = n; size > 1; --size) {

    size_t maxIdx = 0;
    
    for (size_t i = 1; i < size; ++i) {
      if (comp(first[maxIdx], first[i])) maxIdx = i;
    }
    
    if (maxIdx != 0) std::reverse(first, first + maxIdx + 1);
    
    std::reverse(first, first + size);  
  }
}

// Bubble Sort
template<typename T, class Comp>
void bubbleSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  for (size_t i = 0; i < n - 1; i++) {
    bool swapped = false;
    for (size_t j = 0; j < n - i - 1; j++) {
      if (comp(first[j + 1], first[j])) {
        std::swap(first[j], first[j + 1]);
        swapped = true;      
      } 
    }
    if (!swapped) break;
  }
}

// Gnome Sort
template <typename T, class Comp>
void gnomeSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  if (n <= 1) return;
  
  size_t i = 1;
  
  while (i < n) {
    if (!comp(first[i], first[i - 1])) i++;
    else
    {
      std::swap(first[i], first[i - 1]);
      if (i > 1) i--;
      else i++;  
    }
  }
}

// Odd Even Sort
template<typename T, class Comp>
void oddEvenSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  bool sorted = false;
  while (!sorted) {
    sorted = true;
    
    #pragma omp parallel for reduction(&:sorted)
    for (size_t i = 1; i < n; i += 2) {
      if (comp(first[i + 1], first[i])) {
        std::swap(first[i + 1], first[i]);
        sorted = false;
      }
    }
    
    #pragma omp parallel for reduction(&:sorted)
    for (size_t i = 0; i < n; i += 2) {
      if(comp(first[i + 1], first[i])) {
        std::swap(first[i + 1], first[i]);
        sorted = false;
      }
    }
  }
}

// Insertion Sort
template<typename T, class Comp>
void insertionSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  size_t i, key, j;
  
  for (i = 1; i < n; i++) {
    key = first[i];
    j = i - 1;
    
    while (j >= 0 && comp(key, first[j])) {
    
      first[j + 1] = first[j];
      j--;   
    }
    first[j + 1] = key; 
  }
}

// Shell Sort
template <typename T, class Comp>
void shellSort(T* first, T* last, const Comp& comp) {

  size_t n = last - first;
  if (n <= 1) return;
  
  for (size_t gap = n / 2; gap > 0; gap /= 2) {
  
    #pragma omp parallel for
    for (size_t start = 0; start < gap; start++) {
      for (size_t j = start + gap; j < n; j += gap) {
      
        T tmp = first[j];
        size_t k = j;
        
        while (k >= gap && comp(tmp, first[k - gap])) {
          first[k] = first[k - gap];
          k -= gap;
        }
        first[k] = tmp;
      }
    }
  }
}

// Radix Sort
template<typename T, class Comp>
void radixSort(T* first, T* last, const Comp& comp) {
    size_t n = last - first;
    if (n <= 1) return;

    T maxVal = first[0];
    for (size_t i = 1; i < n; i++) {
        if (comp(maxVal, first[i])) maxVal = first[i];
    }

    int NOP = 0;
    T tmp = maxVal;
    while (tmp > 0) {
        NOP++;
        tmp /= 10;
    }

    T divisor = 1;
    std::vector<T> output(n);

    for (int pass = 0; pass < NOP; pass++) {

        int numThreads = omp_get_max_threads();
        std::vector<std::vector<size_t>> local_counts(numThreads, std::vector<size_t>(10, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for
            for (size_t i = 0; i < n; i++) {
                int digit = (first[i] / divisor) % 10;
                local_counts[tid][digit]++;
            }
        }

        std::vector<size_t> offsets(10, 0);
        for (int d = 1; d < 10; d++) {
            size_t sum = 0;
            for (int t = 0; t < numThreads; t++) sum += local_counts[t][d-1];
            offsets[d] = offsets[d-1] + sum;
        }

        std::vector<std::vector<size_t>> thread_offsets(numThreads, std::vector<size_t>(10, 0));
        for (int d = 0; d < 10; d++) {
            size_t sum = offsets[d];
            for (int t = 0; t < numThreads; t++) {
                thread_offsets[t][d] = sum;
                sum += local_counts[t][d];
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<size_t> pos = thread_offsets[tid];
            
            #pragma omp for
            for (size_t i = 0; i < n; i++) {
                int digit = (first[i] / divisor) % 10;
                output[pos[digit]++] = first[i];
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            first[i] = output[i];
        }
        divisor *= 10;
    }
}

// Comb Sort
template<typename T, class Comp>
void combSort(T* first, T* last, const Comp& comp) {
    size_t n = last - first;
    if (n <= 1) return;

    const float shrink = 1.3f;
    size_t gap = n;
    bool sorted = false;

    while (!sorted) {
        gap = static_cast<size_t>(gap / shrink);
        if (gap <= 1) {
            gap = 1;
            sorted = true;
        }

        bool swapped_global = false;

        #pragma omp parallel
        {
            bool swapped_local = false;

            size_t limit = n - gap;
            #pragma omp for
            for (size_t i = 0; i < limit; i++) {
                size_t j = i + gap;
                if (comp(first[j], first[i])) {
                    std::swap(first[i], first[j]);
                    swapped_local = true;
                }
            }

            if (swapped_local) {
                #pragma omp atomic write
                swapped_global = true;
            }
        }

        if (swapped_global)
            sorted = false;
    }
}

// Heap Sort
template<typename T, class Comp>
void heapify(T* first, size_t n, size_t i, const Comp& comp) {
  size_t largest = i;
  size_t left = 2 * i + 1;
  size_t right = 2 * i + 2;
  
  if (left < n && comp(first[largest], first[left])) largest = left;
  if (right < n && comp(first[largest], first[right])) largest = right;
  
  if (largest != i) {
    std::swap(first[i], first[largest]);
    heapify(first, n, largest, comp);  
  }
}

template<typename T, class Comp>
void heapSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  if (n <= 1) return;
  
  for (long long i = (long long)n / 2 - 1; i >= 0; i--) heapify(first, n, (size_t)i, comp);
  
  for (size_t i = n - 1; i > 0; i--) {
    std::swap(first[0], first[i]);
    heapify(first, i, 0, comp);
  }
}

template<typename T, class Comp>
void bucketSort(T* first, T* last, const Comp& comp) {
  size_t n = last - first;
  if (n <= 1) return;
  
  T minVal = *std::min_element(first, last);
  T maxVal = *std::max_element(first, last);
  
  #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
  for (size_t i = 1; i < n; i++) {
    if (first[i] < minVal) minVal = first[i];
    if (first[i] > maxVal) maxVal = first[i];  
  }
          
  size_t range = maxVal - minVal + 1;
  std::vector<T> buckets(range, 0);
  
  #pragma omp parallel
  {
    std::vector<size_t> local(range, 0);
    
    #pragma omp for nowait
    for (size_t i = 0; i < n; i++) local[first[i] - minVal]++;
    
    #pragma omp critical
    {
      for (size_t v = 0; v < range; v++) buckets[v] += local[v];
    }  
  }
  
  size_t idx = 0;
  for (size_t v = 0; v < range; v++) {
    size_t count = buckets[v];
    while(count-- > 0) first[idx++] = v + minVal;
  }
}

/*===================*/
/*    END MY CODE    */
/*===================*/

template<typename T,class BinaryComparisonOp>
void mergeSort_Merge(T *a,T *a_last, T *b,T *b_last, T *out, BinaryComparisonOp &comp) {
   while (a < a_last && b < b_last) *out++ = comp(*a, *b) ? *a++ : *b++;
   while (a < a_last) *out++ = *a++;
   while (b < b_last) *out++ = *b++;
}

template<typename T>
void mergeSort_Copy(T *a_first,T *a_last,T *b) {
   const size_t len = a_last - a_first;
   T *a = a_first;
   for (size_t i = 0; i < len; ++i) b[i] = a[i];
}

template<typename T, class BinaryComparisonOp>
T * BinarySearch(const T value, T *lo, T *hi, const BinaryComparisonOp &comp) {
   while (lo < hi) {
      T *mid = lo + (hi - lo) / 2;
      if (comp(value, *mid)) hi = mid;
      else lo = mid + 1;
   }
   return hi;
}

template<typename T, class BinaryComparisonOp>
void mergeSort_ParallelMerge(T *a, T *a_last, T *b, T *b_last, T *out, const BinaryComparisonOp &comp, int level = 0, const int maxParallelDepth=8) {
   level++;
   if (a == a_last) {
      mergeSort_Copy(b, b_last, out);
      return;
   }
   if (b == b_last) {
      mergeSort_Copy(a, a_last,out);
      return;
   }
   if ((a_last - a) < (b_last - b)) {
      std::swap(a, b);
      std::swap(a_last, b_last);
   }
   const size_t cutOff = 128;
   if ((a_last - a) < cutOff) mergeSort_Merge(a, a_last, b, b_last, out, comp);
   else {
      T *p = a + (a_last - a) / 2;
      T *q = BinarySearch(*p, b, b_last, comp);
      T *r = out + (p - a) + (q - b);
      *r = *p;
      #pragma omp task shared(comp) if (level < maxParallelDepth)
      mergeSort_ParallelMerge(a, p, b, q, out, comp, level, maxParallelDepth);
      mergeSort_ParallelMerge(p + 1, a_last, q, b_last, r + 1, comp, level, maxParallelDepth);
      #pragma omp taskwait
   }
   return;
}

template<typename T, class BinaryComparisonOp>
void mergeSort_TopDownSplit(T *a_first, T *a_last, T *b_first, const BinaryComparisonOp &comp, int level = 0) {
#ifdef _OPENMP
   const size_t maxParallelDepth = 8;
#else
   const size_t maxParallelDepth = 0;
#endif
   const size_t cutOff = 16;
   const size_t len = a_last - a_first;
#ifdef _OPENMP
   const int thread_id = omp_get_thread_num();
#else
   const int thread_id(0);
#endif
   level++;
   if (len > 1) {
      if(len < cutOff) selectionSort(len, a_first, comp);
      else {
         const size_t half_len = len / 2;
         T *a_middle = a_first + half_len;
         T *b_middle = b_first + half_len;
         T *b_last = b_first + len;
         #pragma omp task shared(comp) if (level < maxParallelDepth)
         mergeSort_TopDownSplit(a_first, a_middle, b_first, comp, level);
         mergeSort_TopDownSplit(a_middle, a_last, b_middle, comp, level);
         #pragma omp taskwait
         if (level < maxParallelDepth) mergeSort_ParallelMerge(a_first, a_middle, a_middle, a_last, b_first, comp);
         else mergeSort_Merge(a_first, a_middle, a_middle, a_last, b_first, comp);
         mergeSort_Copy(b_first,b_last,a_first);
      }
   }
}

template<typename T, class BinaryComparisonOp>
void mergeSort(T *first, T *last, const BinaryComparisonOp &comp) {
   if (first >= last) return;
   const size_t n = last - first;
   T *buffer = NULL;
   Allocate(buffer, n);
   #pragma omp parallel default(shared)
   {
      #pragma omp single
      mergeSort_TopDownSplit(first, last, buffer, comp);
   }
   Deallocate(buffer);
}

template<typename T, class BinaryComparisonOp>
void hybridSort(T *first, T *last, const BinaryComparisonOp &comp) {
   if (first >= last) return;
   int nChunks = 4;
#ifdef _OPENMP
   nChunks = omp_get_max_threads();
#endif
   size_t n = last - first;
   size_t chunkSize = n / nChunks;
   TimerType t0 = getTimeStamp();
   std::vector<size_t> idx(nChunks + 1);
   for (int k = 0; k < nChunks; ++k) idx[k] = k * chunkSize;
   idx[nChunks] = n;
   #pragma omp parallel for
   for (int k = 0; k < nChunks; ++k) {
      T *ptr0 = first + idx[k];
      T *ptr1 = first + idx[k + 1];
      std::sort(ptr0, ptr1, comp);
   }
   TimerType t1 = getTimeStamp();
   TimerType t2 = getTimeStamp();
   T *buf = NULL;
   Allocate(buf, n);
   T *in = first;
   T *out = buf;
   for (int stride = 1; stride < nChunks; stride *= 2) {
      #pragma omp parallel for
      for (int k = 0; k < nChunks; k +=2 * stride) {
         T *k_first = &in[idx[k]];
         T *k_mid = &in[idx[k + stride]];
         T *k_last = &in[idx[k + 2 * stride]];
         T *k_out = &out[idx[k]];
         mergeSort_Merge(k_first, k_mid, k_mid, k_last, k_out, comp);
      }
      std::swap(in, out);
   }
   if (out == first) std::copy(buf, buf + n, first);
   Deallocate(buf);
   TimerType t3 = getTimeStamp();
}

template<typename T>
struct random_value {
   T operator() (void)const;
   T operator() (const T,const T)const;
};

template<> float random_value<float> ::
  operator() (const float lo, const float hi)const { 
    return (float(rand()) / RAND_MAX) * (hi -lo) + lo; 
  }
   
template<> float random_value<float> ::
  operator() (void)const { 
    return random_value<float> :: operator() (0.f, 1.f);
  }
   
template<> double random_value<double> ::
  operator() (const double lo, const double hi)const { 
    return(double(rand()) / RAND_MAX) * (hi - lo) + lo;
  }
   
template<> double random_value<double> ::
  operator() (void)const { 
    return random_value<double> :: operator() (0., 1.);
  }
   
template<> int random_value<int> ::
  operator() (const int lo, const int hi)const { 
    return lo + rand() % (hi - lo + 1); 
  }
   
template<> int random_value<int>::
  operator() (void)const { 
    return rand(); 
  }
   
template<> long random_value<long> ::
  operator() (const long lo, const long hi)const { 
    return lo + rand() % (hi - lo + 1); 
  }
   
template<> long random_value<long> ::
  operator() (void)const { 
    return rand(); 
  }
   
template<> char random_value<char> ::
  operator() (const char _lo, const char _hi)const {
    int lo = int(_lo);
    int hi = int(_hi);
    random_value<int> op;
    return char(op(lo, hi));
  }
  
template<> char random_value<char> ::
  operator() (void)const {
    return random_value<char> :: operator() ('a', 'z');
  }
  
typedef int ValueType;
template<typename ValueType, typename IndexType>
struct IndexSortHelper
{
   typedef ValueType value_type;
   typedef IndexType index_type;
   value_type *val;
   index_type *idx;
   IndexSortHelper(value_type *val, index_type *idx) : val(val), idx(idx) { }
   bool operator() (const IndexType &left, const IndexType &right)const { return val[left]<val[right]; }
};

enum algorithmTagType
{
   stdSortTag,
   selectSortTag,
   mergeSortTag,
   quickSortTag,
   hybridSortTag,
   partitionOnlyTag,
   slowSortTag,
   stoogeSortTag,
   bubbleSortTag,
   pancakeSortTag,
   gnomeSortTag,
   oddEvenSortTag,
   insertionSortTag,
   shellSortTag,
   radixSortTag,
   combSortTag,
   heapSortTag,
   bucketSortTag,
   numberOfAlgorithms
};

std::string getAlgorithmName(int tag) {
   switch (tag) {
      case stdSortTag:       return std::string("std::qsort");
      case selectSortTag:    return std::string("selectSort");
      case mergeSortTag:     return std::string("mergeSort");
      case slowSortTag:      return std::string("slowSort");
      case stoogeSortTag:    return std::string("stoogeSort");
      case pancakeSortTag:   return std::string("pancakeSort");
      case bubbleSortTag:    return std::string("bubbleSort");
      case gnomeSortTag:     return std::string("gnomeSort");
      case oddEvenSortTag:   return std::string("oddEvenSort");
      case insertionSortTag: return std::string("insertionSort");
      case shellSortTag:     return std::string("shellSort");
      case radixSortTag:     return std::string("radixSort");
      case combSortTag:      return std::string("combSort");
      case quickSortTag:     return std::string("quickSort");
      case hybridSortTag:    return std::string("hybridSort");
      case heapSortTag:      return std::string("heapSort");
      case bucketSortTag:    return std::string("bucketSort");
      case partitionOnlyTag: return std::string("partitionOnly");
      default:
         fprintf(stderr,"Unknown sorting tag\n");
         exit(-1);
   }
}

algorithmTagType getAlgorithmTag(int tag) {
   if (tag == int(stdSortTag))             return stdSortTag;
   else if (tag == int(selectSortTag))     return selectSortTag;
   else if (tag == int(mergeSortTag))      return mergeSortTag;
   else if (tag == int(slowSortTag))       return slowSortTag;
   else if (tag == int(stoogeSortTag))     return stoogeSortTag;
   else if (tag == int(pancakeSortTag))    return pancakeSortTag;
   else if (tag == int(bubbleSortTag))     return bubbleSortTag;
   else if (tag == int(gnomeSortTag))      return gnomeSortTag;
   else if (tag == int(oddEvenSortTag))    return oddEvenSortTag;
   else if (tag == int(insertionSortTag))  return insertionSortTag;
   else if (tag == int(shellSortTag))      return shellSortTag;
   else if (tag == int(radixSortTag))      return radixSortTag;
   else if (tag == int(combSortTag))       return combSortTag;
   else if (tag == int(quickSortTag))      return quickSortTag;
   else if (tag == int(hybridSortTag))     return hybridSortTag;
   else if (tag == int(heapSortTag))       return heapSortTag;
   else if (tag == int(bucketSortTag))     return bucketSortTag;
   else if (tag == int(partitionOnlyTag))  return partitionOnlyTag;
   else {
      fprintf(stderr,"Unknown sorting tag %d\n",tag);
      exit(-1);
   }
}

int verbose = 0;
double min_time = 0.1;

template<class Table>
int run_test(const int n, int numTests, const algorithmTagType algorithmTag, Table& table, const bool doIndexSort = false) {
   std::vector<ValueType> _a(n), _b(n);
   std::vector<int> _idx(n);
   auto *a = _a.data(), *b = _b.data();
   auto *idx = _idx.data();
   std::less<ValueType> comp;
   typedef IndexSortHelper<ValueType, int> index_comp_type;
   index_comp_type index_comp(b, idx); {
      srand(1);
      random_value<ValueType> random;
      for (int i = 0; i < n; ++i) a[i] = random(0, n);
   }
   auto index_partition_comp = [&](const int i, const ValueType p) { return comp(b[i],p); };
   const ValueType piv = selectPivot(a, a + n);
   ValueType *mid = NULL;
   int *index_mid = NULL;
   double time_sort = 0;
   int niters = (numTests > 1) ? numTests : 1;
   auto run_kernel = [&]() {
         TimerType t_start = getTimeStamp();
         for (int k = 0; k < niters; ++k) {
            dummy_function(n, (void*)a);
            std::copy(a, a + n, b);
            dummy_function(n, (void*)b);
            if (doIndexSort) {
               for (int i = 0; i < n; ++i) idx[i] = i;
               dummy_function(n, (void*)idx);
               if (algorithmTag == partitionOnlyTag) {
#ifdef _OPENMP
                  index_mid = ParallelPartition(idx, n, piv, index_partition_comp);
#else
                  index_mid = myPartition(idx, idx + n, piv, index_partition_comp);
#endif
                  dummy_function(n / 2, (void*)index_mid);
               }
               else if (algorithmTag == selectSortTag)     selectionSort(n, idx, index_comp);
               else if (algorithmTag == mergeSortTag)      mergeSort(idx, idx + n, index_comp);
               else if (algorithmTag == slowSortTag)       slowSort(idx, idx + n, index_comp);
               else if (algorithmTag == stoogeSortTag)     stoogeSort(idx, idx + n, index_comp);
               else if (algorithmTag == pancakeSortTag)    pancakeSort(idx, idx + n, index_comp);
               else if (algorithmTag == bubbleSortTag)     bubbleSort(idx, idx + n, index_comp);
               else if (algorithmTag == gnomeSortTag)      gnomeSort(idx, idx + n, index_comp);
               else if (algorithmTag == oddEvenSortTag)    oddEvenSort(idx, idx + n, index_comp);
               else if (algorithmTag == insertionSortTag)  insertionSort(idx, idx + n, index_comp);
               else if (algorithmTag == shellSortTag)      shellSort(idx, idx + n, index_comp);
               else if (algorithmTag == radixSortTag)      radixSort(idx, idx + n, index_comp);
               else if (algorithmTag == combSortTag)       combSort(idx, idx + n, index_comp);
               else if (algorithmTag == quickSortTag)      quickSort(idx, idx + n, index_comp);
               else if (algorithmTag == hybridSortTag)     hybridSort(idx, idx + n, index_comp);
               else if (algorithmTag == heapSortTag)       heapSort(idx, idx + n, index_comp);
               else if (algorithmTag == bucketSortTag)     bucketSort(idx, idx + n, index_comp);
               else std::sort(idx, idx +n , index_comp);
               dummy_function(n, (void*)idx);
            }
            else {
               if (algorithmTag == partitionOnlyTag)
               {
#ifdef _OPENMP
                  mid = ParallelPartition(b, n, piv, comp);
#else
# ifdef ENABLE_SIMD
# warning "Calling SIMD Parallel Partitioner"
                  mid = myPartitionSimd(b, b + n, piv, comp);
# else
                  mid = myPartition(b, b + n, piv, comp);
# endif
#endif
                  dummy_function(n / 2, (void*)mid);
               }
               else if (algorithmTag == selectSortTag)      selectionSort(n, b, comp);
               else if (algorithmTag == mergeSortTag)       mergeSort(b, b + n, comp);
               else if (algorithmTag == slowSortTag)        slowSort(b, b + n, comp);
               else if (algorithmTag == stoogeSortTag)      stoogeSort(b, b + n, comp);
               else if (algorithmTag == pancakeSortTag)     pancakeSort(b, b + n, comp);
               else if (algorithmTag == bubbleSortTag)      bubbleSort(b, b + n, comp);
               else if (algorithmTag == gnomeSortTag)       gnomeSort(b, b + n, comp);
               else if (algorithmTag == oddEvenSortTag)     oddEvenSort(b, b + n, comp);
               else if (algorithmTag == insertionSortTag)   insertionSort(b, b + n, comp);
               else if (algorithmTag == shellSortTag)       shellSort(b, b + n, comp);
               else if (algorithmTag == radixSortTag)       radixSort(b, b + n, comp);
               else if (algorithmTag == combSortTag)        combSort(b, b + n, comp);
               else if (algorithmTag == quickSortTag)       quickSort(b, b + n, comp);
               else if (algorithmTag == hybridSortTag)      hybridSort(b, b + n, comp);
               else if (algorithmTag == heapSortTag)        heapSort(b, b + n, comp);
               else if (algorithmTag == bucketSortTag)      bucketSort(b, b + n, comp);
               else std::sort(b, b + n, comp);
               dummy_function(n, (void*)b);
            }
         }
         TimerType t_end = getTimeStamp();
         return getElapsedTime(t_start, t_end);
      };
   auto print_list = [&]() {
         if (verbose && n <= 50) {
            printf("List: ");
            if (algorithmTag == partitionOnlyTag) {
              printf("piv: %s, mid: %lu", std::to_string(piv).c_str(), (doIndexSort) ? std::distance(idx, index_mid) : std::distance(b, mid));
            }
            printf("\n");
            for (int i = 0; i < n; ++i) {
               auto _a = std::to_string(a[i]);
               auto _b = std::to_string((doIndexSort) ? b[idx[i]] : b[i]);
               if (doIndexSort) printf("%4d, %10s, %10s, %4d\n", i, _a.c_str(), _b.c_str(), idx[i]);
               else printf("%4d, %10s, %10s\n", i, _a.c_str(), _b.c_str());
            }
         }
      };
   {
      run_kernel();
      if (algorithmTag == partitionOnlyTag) {
         bool yes = true;
         if (doIndexSort) {
            auto n_mid = std::distance(idx, index_mid);
            for (int i = 0; i < n_mid; ++i) yes &= comp(b[idx[i]], piv);
            for (int i = n_mid; i < n; ++i) yes &= not(comp(b[idx[i]], piv));
         }
         else {
            auto n_mid = std::distance(b, mid);
            for (int i = 0; i < n_mid; ++i) yes &= comp(b[i], piv);
            for (int i = n_mid; i < n; ++i) yes &= not(comp(b[i], piv));
         }
         if (not(yes)) {
            fprintf(stderr,"Error: list is not partitioned\n");
            print_list();
            return 1;
         }
      }
      else {
         auto yes = (doIndexSort)
                      ?myIsSorted(idx, idx + n, index_comp)
                      :myIsSorted(b, b + n, comp);
         if (not(yes)){
            fprintf(stderr,"Error: list is not sorted\n");
            print_list();
            return 1;
         }
      }
   }
   bool warmup = true;
   while(1) {
      time_sort = run_kernel();
      if (numTests or not(warmup)) break;
      else {
         if (time_sort > min_time) warmup = false;
         else {
            if (niters > 10) niters = int(1.05 * (min_time / time_sort) * niters) + 1;
            else niters *= 2;
         }
      }
   }
   time_sort /= niters;
   table.push_back(std::make_pair("N", double(n)));
   table.push_back(std::make_pair("Time (ms)", 1000. * time_sort));
   auto memsize = n * sizeof(ValueType);
   if (doIndexSort) memsize += n * sizeof(int);
   if (algorithmTag == mergeSortTag) memsize += (doIndexSort) ? n * sizeof(int) : n * sizeof(ValueType);
   table.push_back(std::make_pair("Size (kb)", double(memsize) / 1024.));
   table.push_back(std::make_pair("Tests", niters));
#ifdef WITH_PAPI
   std::vector<int> papi_events{ PAPI_TOT_CYC, PAPI_L1_DCM, PAPI_L2_TCM, PAPI_L3_TCM, PAPI_TOT_INS, PAPI_VEC_SP };
   std::vector<std::string> papi_event_names;
   for (int i = 0; i < papi_event_names.size(); ++i) {
      int this_event = 0 | PAPI_NATIVE_MASK;
      int retval = PAPI_event_name_to_code(const_cast<char*>(papi_event_names[i].c_str()), &this_event);
      if(retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_event_code_to_name %d\n", retval);
         return 1;
      }
      papi_events.push_back(this_event);
   }

   const int num_papi_events = papi_events.size();
   std::vector<long long> papi_event_counters(num_papi_events);
   const int num_hw_counters = 1;
   for (int event = 0; event < num_papi_events; event += num_hw_counters) {
      int n_events = std::min(num_papi_events - event, num_hw_counters);
      run_kernel();
      int retval = PAPI_start_counters(&papi_events[event], n_events);
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_start_counters %d\n", retval);
         return 1;
      }
      run_kernel();
      retval = PAPI_stop_counters(&papi_event_counters[event], n_events);
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI: Error calling PAPI_stop_counters %d\n", retval);
         return 1;
      }
      for (int i = event; i < (event + n_events); ++i) {
         char papi_event_str[PAPI_MAX_STR_LEN];
         retval = PAPI_event_code_to_name(papi_events[i], papi_event_str);
         if (retval != PAPI_OK) {
            fprintf(stderr,"PAPI: Error calling PAPI_event_code_to_name %d\n", retval);
            return 1;
         }
         double count = double(papi_event_counters[i]) / niters;
         std::string short_name(papi_event_str);
         table.push_back(std::make_pair(short_name, count));
      }
   }
#endif
   printf("finished %d\n",n);
   print_list();
   return 0;
}

int main(int argc, char* argv[]) {
   int minSize = 50, maxSize = 1000000, num_tests = 0, algorithm = 1, doIndexSort = 0;
   float stepSize = 1.5;
   algorithmTagType algorithmTag = stdSortTag;
   for (int i = 1; i < argc; ++i) {
#define check_index(i,str) \
   if ((i)>=argc) \
      {fprintf(stderr,"Missing 2nd argument for %s\n", str);return 1;}
#define print_help() \
      { fprintf(stderr,"sort --help|-h --nelems|-n # -min # -max # --ntests|-t # --algorithm|-a # --index|-i --stepsize|-s #\n"); \
        fprintf(stderr,"\talgorithms: default=0\n"); \
        for(int k=0;k<numberOfAlgorithms;++k) \
           fprintf(stderr,"\t%d) %s\n", k, getAlgorithmName(getAlgorithmTag(k)).c_str()); \
      }
      std::string key(argv[i]);
      if (key == "-h" || key == "--help") {
         print_help();
         return 1;
      }
      else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose++;
      else if (strcmp(argv[i], "--nelems") == 0 || strcmp(argv[i], "-n") == 0) {
         check_index(i + 1, "--nelems|-n");
         i++;
         if (isdigit(*argv[i])) minSize = maxSize =atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-min") == 0) {
         check_index(i+1,"-min");
         i++;
         if (isdigit(*argv[i])) minSize = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "-max") == 0) {
         check_index(i + 1, "-max");
         i++;
         if (isdigit(*argv[i])) maxSize = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "--ntests") == 0 || strcmp(argv[i], "-t") == 0) {
         check_index(i + 1, "--ntests|-t");
         i++;
         if (isdigit(*argv[i]) || *argv[i] == '-') num_tests = atoi(argv[i]);
      }
      else if (strcmp(argv[i], "--algorithm") == 0 || strcmp(argv[i], "-a") == 0) {
         check_index(i + 1, "--algorithm|-a");
         i++;
         if (isdigit(*argv[i])) algorithmTag = getAlgorithmTag(atoi(argv[i]));
      }
      else if (strcmp(argv[i], "--stepsize") == 0 || strcmp(argv[i], "-s") == 0) {
         check_index(i + 1, "--stepsize|-s");
         i++;
         if (isdigit(*argv[i]) || *argv[i] == '.') stepSize = atof(argv[i]);
      }
      else if (key == "--mintime") {
         check_index(i + 1, "--mintime");
         i++;
         min_time = atof(argv[i]);
      }
      else if (strcmp(argv[i], "--index") == 0 || strcmp(argv[i], "-i") == 0) doIndexSort = true;
      else {
         fprintf(stderr, "Unknown option %s\n", argv[i]);
         print_help();
         return 1;
      }
   }
   fprintf(stderr, "algorithm = %d %s\n", algorithmTag, getAlgorithmName(algorithmTag).c_str());
#ifdef _OPENMP
   fprintf(stderr, "num_threads = %d\n", omp_get_max_threads());
#endif
#ifdef WITH_PAPI
   if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
      fprintf(stderr, "PAPI: version mismatch!\n");
      return 1;
   }
   if (PAPI_num_counters() < 2) {
      fprintf(stderr, "PAPI: no hardware counters available!\n");
      return 1;
   }
#endif
   fprintf(stderr, "minSize = %d, maxSize = %d, stepSize = %f, numTests = %d, indexSort = %d, minTime = %f, ValueType=%s\n", minSize, 
                                                                                                                             maxSize, 
                                                                                                                             stepSize, 
                                                                                                                             num_tests, 
                                                                                                                             doIndexSort,
                                                                                                                             min_time,
                                                                                                                             typeid(ValueType).name());
   typedef std::vector<std::pair<std::string,double>> Table;
   auto print_table = [&](const std::vector<Table>& table) {
         std::vector<int> widths;
         if (widths.empty()) {
            auto header = table.front();
            for (int j = 0; j <header.size(); ++j) {
               auto width = std::max(size_t(10), header[j].first.length() + 1);
               widths.push_back(width);
               std::cout << std::setw(width) << header[j].first;
               if (j == header.size() - 1) std::cout << std::endl;
               else std::cout << ",";
            }
         }
         for (int i = 0; i < table.size(); ++i) {
            const auto& row = table[i];
            for (int j = 0; j < row.size(); ++j) {
               auto width = widths[j];
               std::stringstream ss; ss << row[j].second;
               auto s = std::string(ss.str());
               if (s.length() > width) s.resize(width);
               else for (int i = s.length(); i < width; ++i) s.insert(0, " ");
               std::cout << s;
               if (j == row.size() - 1) std::cout << std::endl;
               else std::cout << ",";
            }
         }
      };
   std::vector<Table> table;
   for (int size = minSize; size <= maxSize; size *= stepSize) {
      Table row;
      run_test(size, num_tests, algorithmTag, row, false);
 //   run_test(size,num_tests,algorithmTag,row,doIndexSort);
      table.push_back(row);
   }
   print_table(table);
#ifdef WITH_PAPI
   PAPI_shutdown();
#endif
   return 0;
}