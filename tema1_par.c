#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

typedef struct {
    unsigned char **grid;
    ppm_image *image;
    ppm_image **contour_map;
    ppm_image *new_image;
    pthread_barrier_t *barrier1;
    pthread_barrier_t *barrier2;
    pthread_barrier_t *barrier3;
    unsigned int NUM_THREADS;
} Thread_data;

typedef struct {
    int id;
    Thread_data *thread_data;
} Thread;

int min(int a, int b) {
    return (a < b) ? a : b;
}

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {

       ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);

    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
       for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
unsigned char **sample_grid(ppm_image *image, unsigned char **grid, int step_x, int step_y, unsigned char sigma, int id, unsigned int NUM_THREADS) {

    int p = image->x / step_x;
    int q = image->y / step_y;

    int start_i = id * (double) p / NUM_THREADS;
    int end_i = min((id + 1) * (double) p / NUM_THREADS, p);

    for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = 0; i < p; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }
    for (int j = 0; j < q; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }

    return grid;
}


// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y, int id, unsigned int NUM_THREADS) {
   int p = image->x / step_x;
    int q = image->y / step_y;
    int start_i = id * (double) p / NUM_THREADS;
    int end_i = min((id + 1) * (double) p / NUM_THREADS, p);

    for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(Thread_data *thread_data)  {

    unsigned char **grid = thread_data->grid;
    ppm_image *new_image = thread_data->new_image;
    ppm_image **contour_map = thread_data->contour_map;
    
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= new_image->x / STEP; i++) {
        free(grid[i]);
    }
    free(grid);

    free(new_image->data);
    free(new_image);
    pthread_barrier_destroy(thread_data->barrier1);
    pthread_barrier_destroy(thread_data->barrier2);
    pthread_barrier_destroy(thread_data->barrier3);

}

// Function to rescale an image
ppm_image *rescale_image(ppm_image *image, ppm_image *new_image, int id, unsigned int NUM_THREADS) {
    
    uint8_t sample[3];

    //  we only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        return image;
    }
   
    int  start_i = id * (double) new_image->x / NUM_THREADS;
    int end_i = min((id + 1) * (double) new_image->x / NUM_THREADS, new_image->x);

    // use bicubic interpolation for scaling
      for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < new_image->y; j++) {
            float u = (float)i / (float)(new_image->x - 1);
            float v = (float)j / (float)(new_image->y - 1);
            sample_bicubic(image, u, v, sample);

            new_image->data[i * new_image->y + j].red = sample[0];
            new_image->data[i * new_image->y + j].green = sample[1];
            new_image->data[i * new_image->y + j].blue = sample[2];
        }
    }


    return new_image;
}

void *thread_main(void *arg) {
    Thread thread = *(Thread *)arg;

    int id = thread.id;
    int step_x = STEP;
    int step_y = STEP;
    unsigned int NUM_THREADS = thread.thread_data->NUM_THREADS;

    pthread_barrier_t *barrier1 = thread.thread_data->barrier1;
    pthread_barrier_t *barrier2 = thread.thread_data->barrier2;
    pthread_barrier_t *barrier3 = thread.thread_data->barrier3;


    // 1. Rescale the image
   thread.thread_data->new_image = rescale_image(thread.thread_data->image, thread.thread_data->new_image, id, NUM_THREADS);
   pthread_barrier_wait(barrier1);

    
    // 2. Sample the grid
    thread.thread_data->grid = sample_grid(thread.thread_data->new_image,  thread.thread_data->grid, STEP, STEP, SIGMA, id, NUM_THREADS);
   pthread_barrier_wait(barrier2);
  

    // 3. March the squares
   march( thread.thread_data->new_image, thread.thread_data->grid,  thread.thread_data->contour_map, STEP, STEP, id, NUM_THREADS);
    pthread_barrier_wait(barrier3);

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <num_threads>\n");
        return 1;
    }

    unsigned int NUM_THREADS = atoi(argv[3]);

    // Initialize barriers
    pthread_barrier_t barrier1;
    pthread_barrier_t barrier2;
    pthread_barrier_t barrier3;
    pthread_barrier_init(&barrier1, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier2, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier3, NULL, NUM_THREADS);

    // Initialize thread data
    Thread_data thread_data;
    thread_data.barrier1 = &barrier1;
    thread_data.barrier2 = &barrier2;
    thread_data.barrier3 = &barrier3;
    thread_data.image = read_ppm(argv[1]);
    thread_data.NUM_THREADS = NUM_THREADS;
    thread_data.contour_map = init_contour_map();

    int p = thread_data.image->x / STEP;
    int q = thread_data.image->y / STEP;

    // Allocate memory for grid
    thread_data.grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    for (int i = 0; i <= p; i++) {
        thread_data.grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
    }

    // Allocate memory for new image
    thread_data.new_image = (ppm_image *)malloc(sizeof(ppm_image));
    thread_data.new_image->x = RESCALE_X;
    thread_data.new_image->y = RESCALE_Y;
    thread_data.new_image->data = (ppm_pixel *)malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));

    // Create threads
    pthread_t thread_id[NUM_THREADS];
    Thread thread[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        thread[i].id = i;
        thread[i].thread_data = &thread_data;
        pthread_create(&thread_id[i], NULL, thread_main, (void *)&thread[i]);
    }

      for(int i = 0 ; i < NUM_THREADS ; i++){
        pthread_join(thread_id[i], NULL);
    }
    
      write_ppm(thread_data.new_image , argv[2]);
      free_resources(&thread_data);
   

    return 0;
}
