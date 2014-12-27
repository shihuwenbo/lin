#include "lin.h"

// multiply matrix A with matrix B, store result in matrix C
void mm1(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C) {
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_B; j++) {
            C[i*nc_B+j] = 0.0;
            for(size_t k=0; k<nc_A; k++) {
                C[i*nc_B+j] += A[i*nc_A+k]*B[k*nc_B+j];
            }
            
        }
    }
}

// multiply matrix A with matrix B, store result in matrix C
void mm2(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C) {
    for(size_t i=0; i<nr_A; i++) {
        for(size_t j=0; j<nc_B; j++) {
            C[i*nc_B+j] = 0.0;
        }
        for(size_t k=0; k<nc_A; k++) {
            for(size_t j=0; j<nc_B; j++) {
                C[i*nc_B+j] += A[i*nc_A+k]*B[k*nc_B+j];
            }
        }
    }
}

// multiply matrix A with matrix B, store result in matrix C
void mm3(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C) {

    // temporary storage for B's column
    float *B_tmp = (float*) safe_calloc(nr_B, sizeof(float));

    // make the computation
    for(size_t j=0; j<nc_B; j++) {
        for(size_t k=0; k<nr_B; k++) {
            B_tmp[k] = B[k*nc_B+j];
        }
        for(size_t i=0; i<nr_A; i++) {
            C[i*nc_B+j] = 0.0;
            for(size_t k=0; k<nc_A; k++) {
                C[i*nc_B+j] += A[i*nc_A+k]*B_tmp[k];
            }
        }
    }

    // clean up memory
    free(B_tmp);
}


// multiply matrix A with matrix B, store result in matrix C
void mm4(float *A, size_t nr_A, size_t nc_A,
        float *B, size_t nr_B, size_t nc_B,
        float *C, size_t nr_C, size_t nc_C) {

    // define tile size for mm4
    size_t tilesize_r = 5;
    size_t tilesize_c = 3;
    size_t tilesize_m = 7;

    // check for extreme conditions
    tilesize_r = tilesize_r>nr_C ? nr_C : tilesize_r;
    tilesize_c = tilesize_c>nc_C ? nc_C : tilesize_c;
    tilesize_m = tilesize_m>nc_A ? nc_A : tilesize_m;
    
    // compute the number of tiles
    size_t ntiles_r = (nr_C-1)/tilesize_r+1;
    size_t ntiles_c = (nc_C-1)/tilesize_c+1;
    size_t ntiles = ntiles_r*ntiles_c;

    // compute number of col/row tiles for A/B
    size_t ntiles_c_A = (nc_A-1)/tilesize_m+1;
    size_t ntiles_c_B = (nc_B-1)/tilesize_c+1;

    // number of cells in a tile
    size_t ntcells_C = tilesize_r*tilesize_c;
    size_t ntcells_A = tilesize_r*tilesize_m;
    size_t ntcells_B = tilesize_m*tilesize_c;

    // number of cells
    size_t ncells_C = nr_C*nc_C;
    size_t ncells_A = nr_A*nc_A;
    size_t ncells_B = nr_B*nc_B;

    // allocate shared memory
    size_t nshmem = ntcells_C+ntcells_A+ntcells_B;
    float *shmem = (float*) safe_calloc(nshmem, sizeof(float));
    float *ptrc = &shmem[0];
    float *ptra = &shmem[ntcells_C];
    float *ptrb = &shmem[ntcells_C+ntcells_A];
    
    // iterate through tiles
    for(size_t i=0; i<ntiles; i++) {
        
        // initialize shared memory
        for(size_t ii=0; ii<ntcells_C; ii++) {
            ptrc[ii] = 0.0;
        }

        // get tile r and tile c
        size_t ir = i/ntiles_c;
        size_t ic = i%ntiles_c;
        
        // check boundary condition
        size_t ntcbm1 = ntiles_c_B-1;
        size_t ncolb = ic==ntcbm1 ? (nc_B-ntcbm1*tilesize_c) : tilesize_c;
        size_t ncolc = ncolb;

        // iterate through corresponding tiles for A and B
        for(size_t j=0; j<ntiles_c_A; j++) {

            // check boundary condition
            size_t ntcam1 = ntiles_c_A-1;
            size_t ncola = j==ntcam1 ? (nc_A-ntcam1*tilesize_m) : tilesize_m;
            
            // load tiles for A into shared memory
            for(size_t ii=0; ii<ntcells_A; ii++) {

                // within cell idx
                size_t tr = ii/ncola;
                size_t tc = ii%ncola;
               
                // absolute A index
                size_t aidx = (ir*tilesize_r+tr)*nc_A;
                aidx += j*tilesize_m+tc;
                
                // load to shared memory
                if(aidx < ncells_A) {
                    ptra[tr*ncola+tc] = A[aidx];
                }
            }
            
            // load tiles for B into shared memory
            for(size_t ii=0; ii<ntcells_B; ii++) {
                
                // within tile idx
                size_t tr = ii/ncolb;
                size_t tc = ii%ncolb;
                
                // absolute B index
                size_t bidx = (j*tilesize_m+tr)*nc_B;
                bidx += ic*tilesize_c+tc;

                // load to shared memory
                if(bidx < ncells_B) {
                    ptrb[tr*ncolb+tc] = B[bidx];
                }
            }
         
            // make mat mult in shared memory
            for(size_t ii=0; ii<tilesize_r*ncolc; ii++) {
            
                // within tile idex
                size_t tr = ii/ncolc;
                size_t tc = ii%ncolc;

                // vector dot product
                for(size_t jj=0; jj<ncola; jj++){
                    float prod = ptra[tr*ncola+jj]*ptrb[jj*ncolb+tc];
                    ptrc[tr*ncolc+tc] += prod;
                }
            }
        }

        // write from shared memory to global memory
        for(size_t ii=0; ii<ntcells_C; ii++) {
            
            // within tile idx
            size_t tr = ii/ncolc;
            size_t tc = ii%ncolc;
                
            // absolute C index
            size_t cidx = (ir*tilesize_r+tr)*nc_C;
            cidx += ic*tilesize_c+tc;
           
            // write to global memory
            if(cidx < ncells_C) {
                C[cidx] = ptrc[tr*ncolc+tc];
            }
        }
    }
}
