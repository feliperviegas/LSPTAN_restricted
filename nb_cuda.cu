#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <sstream>
#include "io.hpp"
#include "evaluate.h"
#include <sys/time.h>

#define CUDA_CHECK_RETURN(value) { \
               cudaError_t _m_cudaStat = value;\
               if (_m_cudaStat != cudaSuccess) {\
                       fprintf(stderr, "Error %s at line %d in file %s\n",\
                                       cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
                                       exit(1);\
               }}

#define SIZE_TRAIN 256
#define SIZE_CLASS 128

std::string index(int doc, int term){
	stringstream t, d;
	d << doc;
	t << term;
	return d.str() + "-" + t.str();
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void trainning_kernel2(int *freqClassVector, double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *probClasse,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int numDocs, double *modeloNB) {

	int vecs, len, term;
	double freq;
	double prob, nt, maiorProb;
	extern __shared__ double temp[]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		temp[blockDim.x + 1] = (docTestIndexVector[blockIdx.x + 1]
				- docTestIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(temp[blockDim.x + 1] > blockDim.x)
    		temp[blockDim.x + 2] = ceil(temp[blockDim.x + 1] / (double) blockDim.x);
    	else
    		temp[blockDim.x + 2] = 1.0;
		maiorProb = -99999.9;
	}
	__syncthreads();

	vecs = temp[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) temp[blockDim.x + 2];

	for (int c = 0; c < numClasses; c++) {
		if (tid == 0) {
			// partial sum initialization
			temp[blockDim.x + 3] = log((freqClassVector[c] + alpha) / (numDocs + alpha * numClasses));
		}
		__syncthreads();
		for (int b = 0; b < len; b++) { // loop through 'len' segments
			// first, each thread loads data into shared memory
			if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
				temp[tid] = 0.0;
			else {
				term = docTestVector[docTestIndexVector[blockIdx.x] + b * blockDim.x + tid];
				freq = docTestFreqVector[docTestIndexVector[blockIdx.x] + b * blockDim.x + tid];
				prob = (matrixTermFreq[c * numTerms + term] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
				nt = freqTermVector[term] / totalTermFreq;
				prob = lambda * nt + (1.0 - lambda) * prob;
				if(freqTermVector[term] != 0){
					temp[tid] = freq * log(prob);
		        }
		        else{
		          temp[tid] = 0.0;
		        }
			}
			__syncthreads();

			// next, perform binary tree reduction on shared memory
			for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
				if (tid < d)
					temp[tid] += (tid + d) >= vecs ? 0.0 : temp[tid + d];
				__syncthreads();
			}

			// first thread puts partial result into shared memory
			if (tid == 0) {
				temp[blockDim.x + 3] += temp[0];
			}
			__syncthreads();
		}
		// finally, first thread puts result into global memory
		if (tid == 0) {
			modeloNB[blockIdx.x * numClasses + c] = temp[blockDim.x + 3];
			if (c == 0) {
				maiorProb = temp[blockDim.x + 3];
			} else if (temp[blockDim.x + 3] > maiorProb) {
				maiorProb = temp[blockDim.x + 3];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		probClasse[blockIdx.x] = maiorProb;
	}
}

__global__ void super_parent_freq(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses) {

	int sp = blockIdx.x * blockDim.x + threadIdx.x;
	int term;
	double freq;

	if (sp < numTerms) {
		for (int c = 0; c < numClasses; c++)
			totalTermClassSp[c * numTerms + sp] = 0.0;
		for (int d = 0; d < numDocs; d++) {
			int clas = docClassVector[d];
			int inicio = docIndexVector[d];
			int fim = docIndexVector[d + 1];

			for (int t = inicio; t < fim; t++) {
				term = docVector[t];
				freq = docFreqVector[t];
				if (term == sp && freq > 0) {
					for (int t2 = inicio; t2 < fim; t2++) {
						term = docVector[t2];
						freq = docFreqVector[t2];
						if (term != sp){
							totalTermClassSp[clas * numTerms + sp] += freq;
						}
					}
				}
			}
		}
	}
}

__global__ void find_sp_kernel(int *docIndexVector,
		int *docVector, double *docFreqVector, int numClasses,
		int numTerms, int numDocs, int totalTerms, int *hasSp, int sp) {

	int vecs, len, term;
	double freq;
	__shared__ int aux[2]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		aux[0] = (docIndexVector[blockIdx.x + 1] - docIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(aux[0] > blockDim.x)
			aux[1] = ceil(aux[0] / (double) blockDim.x);
		else
			aux[1] = 1.0;
		hasSp[blockIdx.x] = 0;
	}
	__syncthreads();

	vecs = aux[0]; // communicate vecs and len's values to other threads
	len = aux[1];

	for (int b = 0; b < len; b++) { // loop through 'len' segments
		// first, each thread loads data into shared memory
		if ((b * blockDim.x + tid) < vecs){ // check if outside 'vec' boundary
			term = docVector[docIndexVector[blockIdx.x] + b * blockDim.x + tid];
			freq = docFreqVector[docIndexVector[blockIdx.x] + b * blockDim.x + tid];
			if(term == sp && freq > 0){
				hasSp[blockIdx.x] = 1;
			}
		}
		__syncthreads();
	}
}

__global__ void init_tableProb(int numTerms, int numClasses,double* probSp) {

	int termId = blockIdx.x * blockDim.x + threadIdx.x;
	int i;

	if (termId < numTerms) {
		for (i = 0; i < numClasses; i++) {
			probSp[i * numTerms + termId] = 0.0;
		}
	}
}


__global__ void compute_frequency(int *docIndexVector, int *docVector, double *docFreqVector, 
	int *docClassVector, int numClasses, int numTerms, int numDocs, int totalTerms, int *hasSp, 
	int sp, double *probSp) {

	int vecs, len, term, clas;
	double freq;
	__shared__ int aux[2]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;
	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		aux[0] = (docIndexVector[blockIdx.x + 1] - docIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(aux[0] > blockDim.x)
			aux[1] = ceil(aux[0] / (double) blockDim.x);
		else
			aux[1] = 1.0;
		
	}
	__syncthreads();

	vecs = aux[0]; // communicate vecs and len's values to other threads
	len = aux[1];
	clas = docClassVector[blockIdx.x];

	if(hasSp[blockIdx.x] == 1){
		for (int b = 0; b < len; b++) { // loop through 'len' segments
			// first, each thread loads data into shared memory
			if ((b * blockDim.x + tid) < vecs){ // check if outside 'vec' boundary
				term = docVector[docIndexVector[blockIdx.x] + b * blockDim.x + tid];
				freq = docFreqVector[docIndexVector[blockIdx.x] + b * blockDim.x + tid];
				atomicAdd(&(probSp[clas * numTerms + term]), freq);
			}
			__syncthreads();
		}
	}
}

__global__ void super_parent_train(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses,
		double* probSp, int sp, double alpha) {

	int termId = blockIdx.x * blockDim.x + threadIdx.x;
	int i;

	if (termId < numTerms) {	
		for (i = 0; i < numClasses; i++) {
			probSp[i * numTerms + termId] = (probSp[i * numTerms + termId] + alpha)	/ (totalTermClassSp[i * numTerms + sp] + alpha * (double) totalTerms);
		}
	}
}



// __global__ void super_parent_train(int *docIndexVector, int *docVector,
// 		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
// 		int numTerms, int numDocs, int totalTerms, int numClasses,
// 		double* probSp, int sp, double alpha, int *hasSp) {

// 	int termId = blockIdx.x * blockDim.x + threadIdx.x;
// 	int i, d, t;
// 	int term;
// 	double freq;

// 	if (termId < numTerms) {
// 		for (i = 0; i < numClasses; i++) {
// 			probSp[i * numTerms + termId] = 0;
// 		}

// 		//Calculo da Frequencia de um termo dado Super pai e a Classe
// 		for (d = 0; d < numDocs; d++) {
// 			if(hasSp[d] == 1){
// 				int clas = docClassVector[d];
// 				int inicio = docIndexVector[d];
// 				int fim = docIndexVector[d + 1];

// 				//Procurando Super Pai no documento
// 				for (t = inicio; t < fim; t++) {
// 					term = docVector[t];
// 					freq = docFreqVector[t];
// 					if (term == termId && freq > 0 && sp != termId) {
// 						// probSp[clas * numTerms + termId] += freq;
// 						atomicAdd(&(probSp[clas * numTerms + termId]), freq);
// 					}
// 				}
// 			}
// 		}
    	
// 		for (i = 0; i < numClasses; i++) {
// 			probSp[i * numTerms + termId] = (probSp[i * numTerms + termId] + alpha)	/ (totalTermClassSp[i * numTerms + sp] + alpha * (double) totalTerms);
// 		}
// 	}
// }


__global__ void super_parent_predict2(double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *probClassSp,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int sp, double *modeloNB, double *probSp, int *docClassSp, 
		int *freqClassVector, int numDocs) {

	int vecs, len, term;
	double freq;
	double prob, nt, maiorProb;
	int bestClass;
	extern __shared__ double temp[]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		temp[blockDim.x + 1] = (docTestIndexVector[blockIdx.x + 1] - docTestIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(temp[blockDim.x + 1] > blockDim.x)
    		temp[blockDim.x + 2] = ceil(temp[blockDim.x + 1] / (double) blockDim.x);
    	else
    		temp[blockDim.x + 2] = 1.0;
	}
	__syncthreads();

	vecs = temp[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) temp[blockDim.x + 2];
	for (int c = 0; c < numClasses; c++) {
		// //-------------------------MODELO ORIGINAL-----------------------------------------
		if(tid == 0){
			temp[blockDim.x + 3] = log((freqClassVector[c] + alpha) / (numDocs + alpha * numClasses));
		}
		__syncthreads();
		for (int b = 0; b < len; b++) { // loop through 'len' segments			
			if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
				temp[tid] = 0.0;
			else {
				term = docTestVector[docTestIndexVector[blockIdx.x]	+ b * blockDim.x + tid];
				freq = docTestFreqVector[docTestIndexVector[blockIdx.x]	+ b * blockDim.x + tid];
				nt = freqTermVector[term] / totalTermFreq;
				prob = (matrixTermFreq[c * numTerms + term] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
				if((term != sp) && (probSp[c * numTerms + term] > prob)){
				// if((term != sp)  && (probSp[c * numTerms + term] > (alpha / alpha * (double)totalTerms))){
					prob = log(lambda*nt + (1.0 - lambda)*probSp[c * numTerms + term]);
				 }
				else{
					prob = (matrixTermFreq[c * numTerms + term] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
		 			prob = log(lambda * nt + (1.0 - lambda) * prob);	
				}
				if(freqTermVector[term] != 0){
					temp[tid] = freq * prob;
		        }
		        else{
		          temp[tid] = 0.0;
		        }
			}
		//-------------------------------------------------------------------------------
		//----------------------MODELO SIMPLIFICADO--------------------------------------
		// if(tid == 0){
		// 	temp[blockDim.x + 3] = modeloNB[blockIdx.x * numClasses + c];
		// }
		// __syncthreads();
		// for (int b = 0; b < len; b++) { // loop through 'len' segments			
		// 	if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
		// 		temp[tid] = 0.0;
		// 	else {
		// 		term = docTestVector[docTestIndexVector[blockIdx.x] + b * blockDim.x + tid];
		// 		freq = docTestFreqVector[docTestIndexVector[blockIdx.x] + b * blockDim.x + tid];
		// 		prob = (matrixTermFreq[c * numTerms + term] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
		// 		nt = freqTermVector[term] / totalTermFreq;
		// 		prob = log(lambda * nt + (1.0 - lambda) * prob);
		// 		prob = log(lambda*nt + (1.0 - lambda)*probSp[c * numTerms + term]) - prob;	
		// 		if(freqTermVector[term] != 0 && sp != term){
		// 			temp[tid] = freq * prob;
		//         }
		//         else{
		//           temp[tid] = 0.0;
		//         }
		// 	}
		//---------------------------------------------------------------------------------
			__syncthreads();
			// next, perform binary tree reduction on shared memory
			for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
				if (tid < d)
					temp[tid] += (tid + d) >= vecs ? 0.0 : temp[tid + d];
				__syncthreads();
			}
			// first thread puts partial result into shared memory
			if (tid == 0) {
				temp[blockDim.x + 3] += temp[0];
			}
			__syncthreads();
		}
		// finally, first thread puts result into global memory
		if (tid == 0) {
			if (c == 0) {
				maiorProb = temp[blockDim.x + 3];
				bestClass = c;
			} 
			else if (temp[blockDim.x + 3] > maiorProb) {
				maiorProb = temp[blockDim.x + 3];
				bestClass = c;
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		probClassSp[blockIdx.x] = maiorProb;
		docClassSp[blockIdx.x] = bestClass;
	}
}


extern "C" {

double nb_gpu(const char* filenameTreino, const char* filenameTeste,
		int numDocs, int numClasses, int numTerms, int numDocsTest,
		int numTermsTest, double alpha, double lambda, int cudaDevice) {


	cudaDeviceReset();
	cudaSetDevice(cudaDevice);
	// clock_t begin, endT, end;
	double iTreino, fTreino;
	iTreino = get_wall_time();

	cerr << "Parametros " << alpha << " " << lambda << endl;

	int block_size, n_blocks;
	int *docTestIndexVector = (int*) malloc((numDocsTest + 1) * sizeof(int)); //Alterei numDocs para numDocsTest
	int *docTestVector = NULL;
	double *docTestFreqVector = NULL;
	int *docClassVector = (int*) malloc(numDocs * sizeof(int));

	int *freqClassVector = (int*) malloc(numClasses * sizeof(int));
	double *totalFreqClassVector = (double*) malloc(
			numClasses * sizeof(double));
	double *matrixTermFreq = (double*) malloc(
			(numTerms * numClasses) * sizeof(double));
	double *freqTermVector = (double*) malloc((numTerms) * sizeof(double));
	double totalTermFreq = 0.0;
	int totalTerms = 0;
	
	map<string, int> docAttribute;

	for (int i = 0; i < numClasses; i++) {
		totalFreqClassVector[i] = 0.0;
		freqClassVector[i] = 0;
		for (int j = 0; j < numTerms; j++) {
			matrixTermFreq[i * numTerms + j] = 0.0;
		}
	}
	for (int j = 0; j < numTerms; j++) {
		freqTermVector[j] = 0.0;
	}

	int *docIndexVector = (int*) malloc((numDocs + 1) * sizeof(int));
	int *docVector = NULL;
	double *docFreqVector = NULL;

	set<int> vocabulary;
	docVector = readTrainDataSP(filenameTreino, docIndexVector,
			totalFreqClassVector, freqClassVector, freqTermVector,
			&totalTermFreq, numClasses, numTerms, &totalTerms, matrixTermFreq,
			vocabulary, &docFreqVector, docClassVector);

	double *matrixTermFreq_D;
	cudaMalloc((void **) &matrixTermFreq_D,
			sizeof(double) * (numTerms * numClasses));
	cudaMemcpy(matrixTermFreq_D, matrixTermFreq,
			sizeof(double) * (numTerms * numClasses), cudaMemcpyHostToDevice);


	int *freqClassVector_D;
	cudaMalloc((void **) &freqClassVector_D, sizeof(int) * numClasses);
	cudaMemcpy(freqClassVector_D, freqClassVector, sizeof(int) * numClasses,
			cudaMemcpyHostToDevice);
	double *totalFreqClassVector_D;
	cudaMalloc((void **) &totalFreqClassVector_D, sizeof(double) * numClasses);
	cudaMemcpy(totalFreqClassVector_D, totalFreqClassVector,
			sizeof(double) * numClasses, cudaMemcpyHostToDevice);


	double *freqTermVector_D;
	cudaMalloc((void **) &freqTermVector_D, sizeof(double) * numTerms);
	cudaMemcpy(freqTermVector_D, freqTermVector, sizeof(double) * numTerms,
			cudaMemcpyHostToDevice);

	/* ============================ TESTE ================================*/
	int *realClass = (int*) malloc((numDocsTest + 1) * sizeof(int));

	docTestVector = readTestData(filenameTeste, docTestIndexVector, realClass,
			&docTestFreqVector, numTerms, docAttribute);

	int *docTestIndexVector_D;
	cudaMalloc((void **) &docTestIndexVector_D,
			sizeof(int) * (numDocsTest + 1));
	cudaMemcpy(docTestIndexVector_D, docTestIndexVector,
			sizeof(int) * (numDocsTest + 1), cudaMemcpyHostToDevice);
	int *docTestVector_D;
	cudaMalloc((void **) &docTestVector_D,
			sizeof(int) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestVector_D, docTestVector,
			sizeof(int) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);
	double *docTestFreqVector_D;
	cudaMalloc((void **) &docTestFreqVector_D,
			sizeof(double) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestFreqVector_D, docTestFreqVector,
			sizeof(double) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);

	double *probClasse = (double*) malloc((numDocsTest) * sizeof(double));
	double *probClasse_D;
	cudaMalloc((void **) &probClasse_D, sizeof(double) * (numDocsTest));

	double* modeloNB = (double*) malloc(
			(numClasses * (numDocsTest)) * sizeof(double));
	double* modeloNB_D;
	cudaMalloc((void **) &modeloNB_D,
			sizeof(double) * (numClasses * (numDocsTest)));


	block_size = SIZE_CLASS;
	n_blocks = numDocsTest;
	trainning_kernel2<<<n_blocks, block_size, (block_size + 3) * sizeof(double)>>>(
			freqClassVector_D, matrixTermFreq_D, totalFreqClassVector_D,
			docTestIndexVector_D, docTestVector_D, docTestFreqVector_D,
			probClasse_D, numClasses, numTerms, numDocsTest, freqTermVector_D,
			totalTermFreq, totalTerms, lambda, alpha, numDocs, modeloNB_D);
	

	cudaMemcpy(probClasse, probClasse_D, sizeof(double) * (numDocsTest),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(modeloNB, modeloNB_D,
			sizeof(double) * (numClasses * numDocsTest),
			cudaMemcpyDeviceToHost);

	double valorFinal, maiorProb;
	int maiorClasseProb;
	int *predictClass = (int*) malloc((numDocsTest) * sizeof(int));

	for (int d = 0; d < numDocsTest; d++) {
		maiorProb = modeloNB[d * numClasses + 0];
		maiorClasseProb = 0;
		for (int c = 1; c < numClasses; c++) {
			if (modeloNB[d * numClasses + c] > maiorProb) {
				maiorClasseProb = c;
				maiorProb = modeloNB[d * numClasses + c];
			}
		}
		// cerr << d << " " << maiorProb << " " << maiorClasseProb << endl;
		predictClass[d] = maiorClasseProb;
	}

	// int *correctClass = (int*) malloc(numClasses*sizeof(int));
	// for(int c = 0; c < numClasses; c++) correctClass[c] = 0;
	// cerr << "# Classes Classificadas corretamente\n";
	// for(int d = 0; d < numDocsTest; d++){
	// 	if(predictClass[d] == realClass[d]) correctClass[realClass[d]] += 1;
	// }

	// for(int c = 0; c < numClasses; c++){
	// 	cerr << c << " " << correctClass[c] << endl;
	// }

	valorFinal = evaluate(realClass, predictClass, numDocsTest, 1);
	cerr << "Resultado Naive Bayes "
			<< evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
			<< evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;

  	cout << "Resultado Naive Bayes "
      << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
      << evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;

	cudaFree(probClasse_D);


	/* ============================ SP-TAN ================================*/

	int *docIndexVector_D;
	cudaMalloc((void **) &docIndexVector_D, (numDocs + 1) * sizeof(int));
	cudaMemcpy(docIndexVector_D, docIndexVector, (numDocs + 1) * sizeof(int),
			cudaMemcpyHostToDevice);
	int *docVector_D;
	cudaMalloc((void **) &docVector_D, sizeof(int) * docIndexVector[numDocs]);
	cudaMemcpy(docVector_D, docVector, sizeof(int) * docIndexVector[numDocs],
			cudaMemcpyHostToDevice);
	double *docFreqVector_D;
	cudaMalloc((void **) &docFreqVector_D,
			sizeof(double) * docIndexVector[numDocs]);
	cudaMemcpy(docFreqVector_D, docFreqVector,
			sizeof(double) * docIndexVector[numDocs], cudaMemcpyHostToDevice);
	int *docClassVector_D;
	cudaMalloc((void **) &docClassVector_D, sizeof(int) * numDocs);
	cudaMemcpy(docClassVector_D, docClassVector, sizeof(int) * numDocs,
			cudaMemcpyHostToDevice);

	free(docIndexVector);
	free(docVector);
	free(docFreqVector);
	free(docClassVector);

	double *totalTermClassSp_D;
	cudaMalloc((void **) &totalTermClassSp_D, sizeof(double) * numClasses * numTerms);
	double *probSp = (double*) malloc(numClasses * numTerms * sizeof(double));
	int sp;
	double *probSp_D;
	cudaMalloc((void **) &probSp_D, sizeof(double) * numClasses * numTerms);

	double *probClassSp = (double*) malloc(numDocsTest * sizeof(double));
	double *probClassSp_D;
	cudaMalloc((void **) &probClassSp_D, sizeof(double) * numDocsTest);

	int *docClassSp = (int*) malloc(numDocsTest * sizeof(int));
	int *docClassSp_D;
	cudaMalloc((void **) &docClassSp_D, sizeof(int) * numDocsTest);

	double *probChildSp = (double*) malloc(
			numTerms * sizeof(double));
	double *probChildSp_D;
	cudaMalloc((void **) &probChildSp_D,
			sizeof(double) * numTerms);


	block_size = 384;
	n_blocks = (numTerms + 1) / block_size
			+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
	super_parent_freq<<<n_blocks, block_size>>>(docIndexVector_D, docVector_D,
			docFreqVector_D, docClassVector_D, totalTermClassSp_D, numTerms,
			numDocs, totalTerms, numClasses);

	// double *totalTermClassSp = (double*) malloc(numClasses*numTerms*sizeof(double));
	// cudaMemcpy(totalTermClassSp, totalTermClassSp_D, sizeof(double)*numClasses*numTerms, cudaMemcpyDeviceToHost);
	// for(int t = 0; t < 100000; t++) cerr << t << " " << totalTermClassSp[0 * numTerms + t] << " " << totalTermClassSp[1 * numTerms + t] << " " << totalTermClassSp[3 * numTerms + t] <<  endl;
	// free(totalTermClassSp);

	int *superParents = (int*) malloc((numDocsTest) * sizeof(int));
	double *probAux = (double*) malloc(numDocsTest * sizeof(double));
	for (int d = 0; d < numDocsTest; d++) {
		probAux[d] = -9999999.0;
		superParents[d] = -1;
	}

	int *hasSp_D;
	cudaMalloc( (void**) &hasSp_D, sizeof(int)*numDocs);
	int * hasSp = (int*) malloc(numDocs*sizeof(int));
	// vocabulary.clear();
	// vocabulary.insert(29);
	// vocabulary.insert(50);

	// double *meanClass = (double*) malloc(numClasses*numDocsTest*sizeof(double));
	// int *denClass = (int*) malloc(numClasses*numDocsTest*sizeof(int));
	// for(int d = 0; d < numDocsTest; d++){
	// 	for(int c = 0; c < numClasses; c++){
	// 		meanClass[d*numClasses + c] = 0.0;
	// 		denClass[d*numClasses + c] = 0;
	// 	}
	// }

	// cout << "Number of attributes " << vocabulary.size() << endl;

	// clock_t b, e;
	// float time;
	// cudaEvent_t start, stop;

	for (set<int>::iterator spIt = vocabulary.begin(); spIt != vocabulary.end(); ++spIt) {
		sp = *spIt;
		
		//Start timer
		double wall0 = get_wall_time();

		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
		// cudaEventRecord(start, 0);
		block_size = SIZE_TRAIN;
		n_blocks = numDocs;
		find_sp_kernel<<<n_blocks, block_size>>>(docIndexVector_D, docVector_D, docFreqVector_D,
				numClasses, numTerms, numDocs, totalTerms, hasSp_D, sp);

		// cudaMemcpy(probSp, probSp_D, sizeof(double)*numTerms*numClasses, cudaMemcpyDeviceToHost);
		// for(int c = 0; c < numClasses; c++) cerr << c << " " << probSp[c * numTerms + 0] << endl;


		// if(sp == 6617){
		// 	cudaMemcpy(hasSp, hasSp_D, sizeof(int)*numDocs, cudaMemcpyDeviceToHost);
		// 	int cont = 0;
		// 	for(int d = 0; d < numDocs; d++){
		// 		cerr << d << " " << hasSp[d] << endl;
		// 		if(hasSp[d] == 1) cont +=1;
		// 	}
		// 	cerr << cont << endl;
		// }

		// cudaEventRecord(stop, 0);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&time, start, stop);
		// cerr << setprecision (10) << "GPU Time [ms] " << time << endl;


		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
		// cudaEventRecord(start, 0);
		block_size = 384;
		n_blocks = (numTerms + 1) / block_size
				+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
		init_tableProb<<<n_blocks, block_size>>>(numTerms, numClasses, probSp_D);

		// cudaMemcpy(probSp, probSp_D, sizeof(double)*numTerms*numClasses, cudaMemcpyDeviceToHost);
		// for(int t = 0; t < numTerms; t++) cerr << t << " " << probSp[0 * numTerms + t] << " " << probSp[3 * numTerms + t] << endl;

		block_size = SIZE_TRAIN;
		n_blocks = numDocs;
		compute_frequency<<<n_blocks, block_size>>>(docIndexVector_D, docVector_D, docFreqVector_D, docClassVector_D,
			numClasses, numTerms, numDocs, totalTerms, hasSp_D, sp, probSp_D);

		// cudaMemcpy(probSp, probSp_D, sizeof(double)*numTerms*numClasses, cudaMemcpyDeviceToHost);
		// for(int c = 0; c < numClasses; c++) cerr << c << " " << probSp[c * numTerms + 0] << endl;

		block_size = 384;
		n_blocks = (numTerms + 1) / block_size
				+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
		super_parent_train<<<n_blocks, block_size>>>(docIndexVector_D,
				docVector_D, docFreqVector_D, docClassVector_D,
				totalTermClassSp_D, numTerms, numDocs, totalTerms, numClasses,
				probSp_D, sp, alpha);

		// cudaEventRecord(stop, 0);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&time, start, stop);
		// cerr << setprecision (10) << "GPU Time [ms] " << time << endl;

		// cudaMemcpy(probSp, probSp_D, sizeof(double)*numTerms*numClasses, cudaMemcpyDeviceToHost);
		// for(int t = 0; t < numTerms; t++){
		// 	cerr << t << " ";
			// for(int c = 0; c < numClasses; c++){
			// 	double nt = freqTermVector[t] / totalTermFreq;
			// 	double prob = (matrixTermFreq[c * numTerms + t] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
			// 	prob = log(lambda * nt + (1.0 - lambda) * prob);  
			// 	cerr << c << " " << prob << " ";
			// 	//cerr << c << " " << probSp[c * numTerms + t] << " ";
			// }
			// cerr << endl;
		// }
		
		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
		// cudaEventRecord(start, 0);

		block_size = SIZE_CLASS;
		n_blocks = numDocsTest;
		super_parent_predict2<<<n_blocks, block_size,
				(block_size + 3) * sizeof(double)>>>(matrixTermFreq_D,
				totalFreqClassVector_D, docTestIndexVector_D, docTestVector_D,
				docTestFreqVector_D, probClassSp_D, numClasses, numTerms,
				numDocsTest, freqTermVector_D, totalTermFreq, totalTerms,
				lambda, alpha, sp, modeloNB_D, probSp_D, docClassSp_D, freqClassVector_D, numDocs);


		// cudaEventRecord(stop, 0);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&time, start, stop);
		// cerr << setprecision (10) << "GPU Time [ms] " << time << endl;


		// b=clock();	
	
		//Avaliação da qualidade de classificação dado o Super Pai
		cudaMemcpy(probClassSp, probClassSp_D, numDocsTest * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(docClassSp, docClassSp_D, numDocsTest * sizeof(int),	cudaMemcpyDeviceToHost);

		// if(sp == 0){
		// 	for (int d = 0; d < numDocsTest; d++) {
		// 		cerr << d << " " << probClassSp[d] << " " << docClassSp[d] << endl;
		// 	}
		// }

		for (int d = 0; d < numDocsTest; d++) {
			// if((realClass[d] == 4 && docClassSp[d] == 4) || (realClass[d] == 9 && docClassSp[d] == 9)){
			// 	cerr << d << " " << realClass[d] << " " << sp << " " << probClassSp[d] << " " << probClasse[d]<< endl;
			// }
			// if(docAttribute[d*numTerms+sp] == 1){
			// 	meanClass[d*numClasses + docClassSp[d]] += probClassSp[d];
			// 	denClass[d*numClasses + docClassSp[d]] += 1;
			// }
			// if(docAttribute.find(index(d,sp)) != docAttribute.end()) cerr << "Doc " << d << " " <<  index(d, sp) <<  endl;
			if ((probClassSp[d] > probAux[d]) && (docAttribute.find(index(d, sp)) != docAttribute.end())){
				cerr << " ( " << d+1 << " " << probAux[d] << " -> " << probClassSp[d] << " " << docClassSp[d] << " ) " ; 
				probAux[d] = probClassSp[d];
				superParents[d] = sp;
				predictClass[d] = docClassSp[d];
			}
		}
		// e=clock();
		// cerr << "SP: " << sp << " " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " " << evaluate(realClass, predictClass, numDocsTest, 0) * 100;
		// cerr << " Time " << double(e-b)/CLOCKS_PER_SEC;

		//Stop timers
    	double wall1 = get_wall_time();

		cerr << endl << "SP: " << sp << " " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " " << evaluate(realClass, predictClass, numDocsTest, 0) * 100;
		cerr << " " << wall1 - wall0;
		cerr << endl;
	}
	// CUDA_SAFE_CALL(cudaEventDestroy(start));
 //    CUDA_SAFE_CALL(cudaEventDestroy(stop));
	free(hasSp);
	docAttribute.clear();

	// for(int d=0; d< numDocsTest; d++){
	// 	cerr << d << " " << superParents[d] << " " << probAux[d] << " " << predictClass[d] << endl;
	// }

	// cerr << "# Primeiro Teste\n";
	// for(int d = 0; d < numDocsTest; d++){
	// 	if(realClass[d] == 4 || realClass[d] == 9){
	// 		cerr << d << " ";
	// 		for(int c = 0; c < numClasses; c++){
	// 			if(denClass[d*numClasses + c] != 0)
	// 				cerr << "Classe " << c << " Numerador: " << meanClass[d*numClasses + c] << " Denominador: " << denClass[d*numClasses + c] << " Razao: " <<  (meanClass[d*numClasses + c]) / (denClass[d*numClasses + c]) << " ";
	// 		}	
	// 		cerr << endl;
	// 	}
	// }
	// free(meanClass);
	// free(denClass);

	// cerr << "# Resultado dos SP para as classes 4 e 9\n";
	// for(int d = 0; d < numDocsTest; d++){
	// 	if(realClass[d] == 4 || realClass[d] == 9)
	// 		cerr << d << " " << realClass[d] << " " << predictClass[d] << " " << superParents[d] << " " << probAux[d] << " " << probClasse[d] << endl;
	// }

	// cerr << "TESTE DE SANIDADE\n";
	// for(int d =0; d < numDocsTest; d++){
	// 	cerr << d << " " << predictClass[d] << " " << probAux[d] << endl;
	// }

	// for(int c = 0; c < numClasses; c++) correctClass[c] = 0;
	// cerr << "# Classes Classificadas corretamente\n";
	// for(int d = 0; d < numDocsTest; d++){
	// 	if(predictClass[d] == realClass[d]) correctClass[realClass[d]] += 1;
	// }

	// for(int c = 0; c < numClasses; c++){
	// 	cerr << c << " " << correctClass[c] << endl;
	// }
	// free(correctClass);
	// endT = clock();
	cerr << "Melhor SP " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
			<< evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;
	cout << "Melhor SP " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
			<< evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;


	// ofstream predict("predict.dat");
	// for(int d=0; d < numDocsTest; d++){
	// 	predict << d << " real " << realClass[d] << " predict " << predictClass[d] << endl;
	// }
	// predict.close();
	
	cudaFree(docIndexVector_D);
	cudaFree(docVector_D);
	cudaFree(docFreqVector_D);
	cudaFree(docClassVector_D);
	cudaFree(docTestIndexVector_D);
	cudaFree(docTestVector_D);
	cudaFree(docTestFreqVector_D);
	cudaFree(freqTermVector_D);
	free(freqTermVector);

	free(docClassSp);
	cudaFree(docClassSp_D);
	cudaFree(totalFreqClassVector_D);
	free(totalFreqClassVector);

	cudaFree(matrixTermFreq_D);
	free(matrixTermFreq);

	cudaFree(hasSp_D);
	cudaFree(totalTermClassSp_D);

	cudaFree(probClassSp_D);
	free(probClassSp);

	cudaFree(modeloNB_D);
	free(modeloNB);

	cudaFree(probSp_D);
	free(probSp);

	free(realClass);
	free(predictClass);
	free(probAux);
	free(superParents);
	free(docTestIndexVector);
	free(docTestVector);
	free(docTestFreqVector);

	cudaFree(probChildSp_D);
	free(probChildSp);
	// end = clock();
	fTreino = get_wall_time();
	cerr << "Time " <<  fTreino - iTreino << endl;
  	cout << "Time " <<  fTreino - iTreino << endl;

	return valorFinal;
}
}
