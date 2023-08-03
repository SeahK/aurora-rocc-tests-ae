#ifndef XR_MT_IMAGES_H
#define XR_MT_IMAGES_H
static const elem_t image_handnet[1][128][128][3] row_align(1);
static const elem_t image_rcnnnet[1][160][160][3] row_align(1);
static const elem_t image_fbnet[1][160][160][3] row_align(1);
static const elem_t image_midasnet[1][128][128][32] row_align(MAX_BLOCK_LEN);
static const elem_t image_ritnet[1][160][160][32] row_align(MAX_BLOCK_LEN);
static const elem_t image_tcnnet[1][112][112][192] row_align(MAX_BLOCK_LEN);
static const elem_t image_rgbnet[1][456][456][64] row_align(MAX_BLOCK_LEN);



#endif
