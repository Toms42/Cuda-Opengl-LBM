//Curl calcs/rendering:
__device__
int computeCurlMiddleCase(int col, int row, lbm_node * array1) {
	return (array1[getIndex(row, col + 1)].yvel - array1[getIndex(row, col - 1)].yvel) - (array1[getIndex(row + 1, col)].xvel - array1[getIndex(row - 1, col)].xvel);
}
__device__
int computeCurlEdgeCase(int col, int row, lbm_node * array1) {
	return 2 * (array1[getIndex(row, col)].yvel - array1[getIndex(row, col - 1)].yvel) - (array1[getIndex(row, col)].xvel - array1[getIndex(row - 1, col)].xvel);
}
__device__
void updatePictureCurl(int col, int row, uchar4 * picture, lbm_node * array) {
	//printf("updating dat curl!\n");
	int nColors = 255;
	int contrast = 70;
	int colorIndex;
	if (0 < col && col < W - 1) {
		if (0 < row && row < H - 1) {
			//picture[getIndex(x,y)]
			colorIndex = (int)(nColors * (0.5 + computeCurlMiddleCase(col, row, array) * contrast * 0.3));
		}
		//else {
		//	//picture[getIndex(x,y)]
		//	colorIndex = (int)(nColors * (0.5 + computeCurlEdgeCase(col, row, array) * contrast * 0.3));
		//}
	}

	picture[getIndex(row, col)].z = colorIndex;

}

//what to render switch-case
__device__
void computeColor(int col, int row, uchar4 * picture, int display, lbm_node * before, lbm_node * after) {

	switch (display) {
	case 1: // one is curl
		updatePictureCurl(col, row, picture, before);
	}
}

__global__
void test(lbm_node * before, lbm_node * after, unsigned char * barrier, int display, uchar4 * picture) {
	printf("before: %x\nafter: %x\nbarrier: %x\n display: %d\n", before, after, barrier, display); //, picture);
}

__global__
void collide(lbm_node * before, lbm_node * after, unsigned char * barrier, int display, uchar4 * picture) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int index0 = getIndex(row, col);

	if (col < 0 || col >= W || row < 0 || row >= H)
		return;

	float n, one9thn, one36thn, vx, vy, vx2, vy2, vx3, vy3, vxvy2, v2, v215;

	// particle collision code

	if (index0 < 0) {
		return;
	}

	if (barrier[index0] == 0) {
		n = before[index0].v0 + before[index0].vN + before[index0].vS + before[index0].vE + before[index0].vW + before[index0].vNW + before[index0].vNE + before[index0].vSW + before[index0].vSE;
		before[index0].density = n; // macroscopic density may be needed for plotting
		one9thn = one9th * n;
		one36thn = one36th * n;
		if (n > 0) {
			vx = (before[index0].vE + before[index0].vNE + before[index0].vSE - before[index0].vW - before[index0].vNW - before[index0].vSW) / n;
		}
		else vx = 0;
		before[index0].xvel = vx; // may be needed for plotting
		if (n > 0) {
			vy = (before[index0].vN + before[index0].vNE + before[index0].vNW - before[index0].vS - before[index0].vSE - before[index0].vSW) / n;
		}
		else vy = 0;
		before[index0].yvel = vy; // may be needed for plotting
		vx3 = 3 * vx;
		vy3 = 3 * vy;
		vx2 = vx * vx;
		vy2 = vy * vy;
		vxvy2 = 2 * vx * vy;
		v2 = vx2 + vy2;
		before[index0].speed2 = v2; // may be needed for plotting
		v215 = 1.5 * v2;
		after[index0].v0 = before[index0].v0 + omega * (four9ths * n * (1 - v215) - before[index0].v0);
		after[index0].vE = before[index0].vE + omega * (one9thn * (1 + vx3 + 4.5 * vx2 - v215) - before[index0].vE);
		after[index0].vW = before[index0].vW + omega * (one9thn * (1 - vx3 + 4.5 * vx2 - v215) - before[index0].vW);
		after[index0].vN = before[index0].vN + omega * (one9thn * (1 + vy3 + 4.5 * vy2 - v215) - before[index0].vN);
		after[index0].vS = before[index0].vS + omega * (one9thn * (1 - vy3 + 4.5 * vy2 - v215) - before[index0].vS);
		after[index0].vNE = before[index0].vNE + omega * (one36thn * (1 + vx3 + vy3 + 4.5 * (v2 + vxvy2) - v215) - before[index0].vNE);
		after[index0].vNW = before[index0].vNW + omega * (one36thn * (1 - vx3 + vy3 + 4.5 * (v2 - vxvy2) - v215) - before[index0].vNW);
		after[index0].vSE = before[index0].vSE + omega * (one36thn * (1 + vx3 - vy3 + 4.5 * (v2 - vxvy2) - v215) - before[index0].vSE);
		after[index0].vSW = before[index0].vSW + omega * (one36thn * (1 - vx3 - vy3 + 4.5 * (v2 + vxvy2) - v215) - before[index0].vSW);
	}

	//printf("calling render!\n");
	computeColor(col, row, picture, display, before, after);
}

__global__
void stream(lbm_node * before, lbm_node * after, unsigned char * barrier) {

	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < 1 || col >= W - 1 || row < 1 || row >= H - 1)
		return;

	int index0 = getIndex(row, col);
	// these are the indices which will be used to retrieve the data for the current vectors
	int indexN = getIndex(row - 1, col);
	int indexS = getIndex(row + 1, col);
	int indexE = getIndex(row, col - 1);
	int indexW = getIndex(row, col + 1);
	int indexNE = getIndex(row - 1, col - 1);
	int indexNW = getIndex(row - 1, col + 1);
	int indexSE = getIndex(row + 1, col - 1);
	int indexSW = getIndex(row + 1, col + 1);

	// particle streaming
	if (0 <= row && row < H - 1) {
		if (0 < col < W - 1) {
			after[index0].vN = before[indexN].vN; // move the north-moving particles
			after[index0].vNW = before[indexNW].vNW; // and the northwest-moving particles
			after[index0].vW = before[indexW].vW; // move the west-moving particles
			after[index0].vSW = before[indexSW].vSW; // and the southwest-moving particles
		}
	}
	if (0 < row && row <= H - 1) {
		if (0 < col && col <= W - 1) {
			after[index0].vE = before[indexE].vE; // move the east-moving particles
			after[index0].vNE = before[indexNE].vNE; // and the northeast-moving particles
		}
		if (0 <= col && col < W - 1) {
			after[index0].vS = before[indexS].vS; // move the south-moving particles
			after[index0].vSE = before[indexSE].vSE; // and the southeast-moving particles
		}
	}
	// CHECK THIS CODE IF BARRIERS DO NOT WORK
	if (row == 0) {
		after[index0].vS = before[indexS].vS;
		if (!barrier[index0]) {
			after[index0].vE = one9th * (1 + 3 * v + 3 * v * v);
			after[index0].vNE = one36th * (1 + 3 * v + 3 * v * v);
			after[index0].vSE = one36th * (1 + 3 * v + 3 * v * v);
		}
	}
	if (row == H - 1) {
		after[index0].vN = before[indexN].vN;
		if (!barrier[index0]) {
			after[index0].vW = one9th * (1 - 3 * v + 3 * v * v);
			after[index0].vNW = one36th * (1 - 3 * v + 3 * v * v);
			after[index0].vSW = one36th * (1 - 3 * v + 3 * v * v);
		}
	}
	if (col == 0) {
		after[index0].v0 = four9ths * (1 - 1.5 * v * v);
		after[index0].vE = one9th * (1 + 3 * v + 3 * v * v);
		after[index0].vW = one9th * (1 - 3 * v + 3 * v * v);
		after[index0].vN = one9th * (1 - 1.5 * v * v);
		after[index0].vS = one9th * (1 - 1.5 * v * v);
		after[index0].vNE = one36th * (1 + 3 * v + 3 * v * v);
		after[index0].vSE = one36th * (1 + 3 * v + 3 * v * v);
		after[index0].vNW = one36th * (1 - 3 * v + 3 * v * v);
		after[index0].vSW = one36th * (1 - 3 * v + 3 * v * v);
	}
	else if (col == W - 1) {
		after[index0].v0 = four9ths * (1 - 1.5 * v * v);
		after[index0].vE = one9th * (1 + 3 * v + 3 * v * v);
		after[index0].vW = one9th * (1 - 3 * v + 3 * v * v);
		after[index0].vN = one9th * (1 - 1.5 * v * v);
		after[index0].vS = one9th * (1 - 1.5 * v * v);
		after[index0].vNE = one36th * (1 + 3 * v + 3 * v * v);
		after[index0].vSE = one36th * (1 + 3 * v + 3 * v * v);
		after[index0].vNW = one36th * (1 - 3 * v + 3 * v * v);
		after[index0].vSW = one36th * (1 - 3 * v + 3 * v * v);
	}
}



__global__
void bounce(lbm_node * before, lbm_node * after, unsigned char * barrier) {

	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < 1 || col >= W - 1 || row < 1 || row >= H - 1)
		return;

	int index0 = getIndex(row, col);
	// these are the indices which will be used to retrieve the data for the current vectors
	int indexN = getIndex(row - 1, col);
	int indexS = getIndex(row + 1, col);
	int indexE = getIndex(row, col - 1);
	int indexW = getIndex(row, col + 1);
	int indexNE = getIndex(row - 1, col - 1);
	int indexNW = getIndex(row - 1, col + 1);
	int indexSE = getIndex(row + 1, col - 1);
	int indexSW = getIndex(row + 1, col + 1);

	// BOUNCE STEP
	if (barrier[index0]) {
		if (before[index0].vN > 0) {
			after[indexN].vS = before[indexN].vS + before[index0].vN;
			before[index0].vN = 0;
		}
		if (before[index0].vS > 0) {
			after[indexS].vN = before[indexS].vN + before[index0].vS;
			before[index0].vS = 0;
		}
		if (before[index0].vE > 0) {
			after[indexE].vW = before[indexE].vW + before[index0].vE;
			before[index0].vE = 0;
		}
		if (before[index0].vW > 0) {
			after[indexW].vE = before[indexW].vE + before[index0].vW;
			before[index0].vW = 0;
		}
		if (before[index0].vNW > 0) {
			after[indexNW].vSE = before[indexNW].vSE + before[index0].vNW;
			before[index0].vNW = 0;
		}
		if (before[index0].vNE > 0) {
			after[indexNE].vSW = before[indexNE].vSW + before[index0].vNE;
			before[index0].vNE = 0;
		}
		if (before[index0].vSW > 0) {
			after[indexSW].vNE = before[indexSW].vNE + before[index0].vSW;
			before[index0].vSW = 0;
		}
		if (before[index0].vSE > 0) {
			after[indexSE].vNW = before[indexSE].vNW + before[index0].vSE;
			before[index0].vSE = 0;
		}
	}
}

__device__
unsigned char clip(int n) {
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

//init stuff:

for (int y = 0; y < H; y++) {
	for (int x = 0; x < W; x++) {
		index0 = getIndex_cpu(y, x);
		if (barrier[index0]) {
			zeroSite(before, index0);
		}
		else {
			before[index0].v0 = four9ths * (1 - 1.5 * v * v);
			before[index0].vE = one9th * (1 + 3 * v + 3 * v * v);
			before[index0].vW = one9th * (1 - 3 * v + 3 * v * v);
			before[index0].vN = one9th * (1 - 1.5 * v * v);
			before[index0].vS = one9th * (1 - 1.5 * v * v);
			before[index0].vNE = one36th * (1 + 3 * v + 3 * v * v);
			before[index0].vSE = one36th * (1 + 3 * v + 3 * v * v);
			before[index0].vNW = one36th * (1 - 3 * v + 3 * v * v);
			before[index0].vSW = one36th * (1 - 3 * v + 3 * v * v);
			before[index0].density = 1;
			before[index0].xvel = v;
			before[index0].yvel = 0;
			before[index0].speed2 = v * v;
		}
	}
}


void zeroSite(lbm_node * array1, int index0) {
	array1[index0].v0 = 0;
	array1[index0].vE = 0;
	array1[index0].vW = 0;
	array1[index0].vN = 0;
	array1[index0].vS = 0;
	array1[index0].vNE = 0;
	array1[index0].vNW = 0;
	array1[index0].vSE = 0;
	array1[index0].vSW = 0;
	array1[index0].xvel = 0;
	array1[index0].yvel = 0;
	array1[index0].speed2 = 0;
}
