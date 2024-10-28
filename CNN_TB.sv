// Convolutional Neural Network TB. 
//
// The images are taken from MNIST digits dataset. 
// 
// All weights and reference calculated by "cnn_behind.py"   
//
// -----------------------------------------------------------------------------
// Copyright (c) 2014-2024 All rights reserved
// -----------------------------------------------------------------------------
// Author : Maksim Ananev mananev086@gmail.com
// 
// Create : 2024-05-13 11:30:23
// Revise : 2024-10-22 12:20:46
// Editor : sublime text4, tab size (4)
// -----------------------------------------------------------------------------



`timescale 1ns/1ns

module CNN_TB ();

    parameter CLASSES_QNT = 10;
    parameter IMG_WIDTH   = 28;
    parameter IMG_HEIGHT  = 28;

    real image_7[IMG_HEIGHT][IMG_WIDTH] =
        '{
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0},
            '{0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170,  52,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,  67, 114,  72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  66,  14, 67,  67,  67,  59,  21, 236, 254, 106,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,  22, 233, 255,  83,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0, 129, 254, 238,  44,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,  59, 249, 254,  62,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   9, 205, 248,  58,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 126, 254, 182,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 75, 251, 240,  57,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  19, 221, 254, 166,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 203, 254, 219,  35,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38, 254, 254,  77,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 224, 254, 115,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254, 254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 254, 219,  40,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 207,  18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
        };

    real image_2[IMG_HEIGHT][IMG_WIDTH] =
        '{

            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   116, 125, 171, 255,   255, 150,  93,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   169, 253, 253, 253, 253,   253, 253, 218,  30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   169, 253, 253, 253, 213, 142,   176, 253, 253, 122,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  52, 250, 253, 210,  32,  12,   0, 6, 206, 253, 140,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  77, 251, 210,  25,   0,   0,   0, 122, 248, 253,  65,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,  31,  18,   0,   0,   0,   0, 209, 253, 253,  65,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   117,   247, 253, 198,  10,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  76, 247, 253, 231,  63,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   128, 253,   253, 144,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   176, 246, 253,   159,  12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 234, 253, 233, 35,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   198, 253, 253, 141,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,  78, 248, 253, 189,  12, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,  19, 200, 253, 253, 141,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   134, 253, 253, 173,  12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   248, 253, 253,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   248, 253, 253,  43,  20,  20,   20,  20,   5,   0,   5,  20,  20,  37, 150, 150, 150, 147,  10,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   248, 253, 253, 253, 253, 253,   253, 253, 168, 143, 166, 253, 253, 253, 253, 253, 253, 253, 123,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   174, 253, 253, 253, 253, 253,   253, 253, 253, 253, 253, 253, 249, 247, 247, 169, 117, 117,  57,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   118, 123, 123, 123, 166,   253, 253, 253, 155, 123, 123,  41,   0,   0,   0,   0,   0,   0,   0},

            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0}

        };
    real image_1[IMG_HEIGHT][IMG_WIDTH] =
        '{  '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,  38, 254, 109,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,  87, 252,  82,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   135, 241,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,  45, 244, 150,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,  84, 254,  63,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   202, 223,  11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 32, 254, 216,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 95, 254, 195,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 140, 254,  77,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  57, 237, 205,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   124, 255, 165,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   171,   254,  81,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 232, 215,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   120, 254,   159,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   151, 254,   142,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   228, 254,   66,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 251, 254, 66,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   141, 254, 205,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  10, 215, 254, 121, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5, 198, 176,  10, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0}
        };


    real image_0[IMG_HEIGHT][IMG_WIDTH] =
        '{
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 150, 253, 202,  31,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  37, 251, 251, 253, 107,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  21, 197, 251, 251, 253, 107,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   110, 190, 251, 251, 251, 253, 169, 109,  62,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   253, 251, 251, 251, 251, 253, 251, 251, 220,  51,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   182, 255, 253, 253, 253, 253, 234, 222, 253, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,  63, 221, 253, 251, 251, 251, 147, 77,  62, 128, 251, 251, 105,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  32, 231, 251, 253, 251, 220, 137, 10,  0,   0,  31, 230, 251, 243, 113,   5,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  37, 251, 251, 253, 188,  20,   0, 0,  0,   0,   0,   109, 251, 253, 251,  35,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  37, 251, 251, 201,  30,   0,   0, 0,  0,   0,   0,  31, 200, 253, 251,  35,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  37, 253, 253,   0,   0,   0,   0, 0,  0,   0,   0,  32, 202, 255, 253, 164,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   140, 251, 251,   0,   0,   0,   0,  0,   0,   0,   0,   109, 251, 253, 251,  35,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   217, 251, 251,   0,   0,   0,   0,  0,   0,  21,  63, 231, 251, 253, 230,  30,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   217, 251, 251,   0,   0,   0,   0,  0,   0,   144, 251, 251, 251, 221,  61,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   217, 251, 251,   0,   0,   0,   0,  0,   182, 221, 251, 251, 251, 180,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   218, 253, 253,  73,  73, 228, 253,  253, 255, 253, 253, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   113, 251, 251, 253, 251, 251, 251,  251, 253, 251, 251, 251, 147,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,  31, 230, 251, 253, 251, 251, 251,  251, 253, 230, 189,  35,  10,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,  62, 142, 253, 251, 251, 251,  251, 253, 107,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  72, 174, 251, 173,  71,  72,  30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},

            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0},
            '{0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0}
        };


    parameter PIX_WIDTH    = 16;
    parameter WEIGHT_WIDTH = 16;
    parameter FRACT_WIDTH  = 12;
    parameter CONV_NUMB    = 2 ;
    parameter FLAT_NUMB    = 2 ;

    parameter logic [CONV_NUMB-1:0][ 1:0][7:0] CONV_DIMENSION   = {{8'd8, 8'd4}, {8'd4, 8'd1}};
    parameter logic [CONV_NUMB-1:0][ 3:0]      KERNEL_DIMENSION = {4'd3  , 4'd3}              ;
    parameter logic [FLAT_NUMB-1:0][15:0]      FLAT_DIMENSION   = {16'd64, 16'd200}           ;



    logic clk    = 0;
    logic clk_en = 1;
    logic rst_n  = 0;

    logic [PIX_WIDTH-1:0] i_data;

    logic i_valid = 0;
    logic i_sop   = 0;
    logic i_eop   = 0;

    logic o_valid;

    logic [CLASSES_QNT-1:0][31:0] classes;

    localparam R2I_COEF = 2**FRACT_WIDTH;

    CNN #(
        .PIX_WIDTH       (PIX_WIDTH       ),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH    ),
        .FRACT_WIDTH     (FRACT_WIDTH     ),
        .CONV_NUMB       (CONV_NUMB       ),
        .CONV_DIMENSION  (CONV_DIMENSION  ),
        .KERNEL_DIMENSION(KERNEL_DIMENSION),
        .FLAT_NUMB       (FLAT_NUMB       ),
        .FLAT_DIMENSION  (FLAT_DIMENSION  ),
        .CLASSES_QNT     (CLASSES_QNT     ),
        .IMG_WIDTH       (IMG_WIDTH       ),
        .IMG_HEIGHT      (IMG_HEIGHT      )
    ) inst_CNN (
        .clk    (clk    ),
        .clk_en (clk_en ),
        .rst_n  (rst_n  ),
        .i_data (i_data ),
        .i_valid(i_valid),
        .i_sop  (i_sop  ),
        .i_eop  (i_eop  ),
        .o_valid(o_valid),
        .classes(classes)
    );


    initial begin
        forever begin
            #10 clk = !clk;
        end
    end

    initial begin
        #100;

        rst_n = 1;

        #100;

        foreach (image_0[row,col]) begin
            @(posedge clk);
            i_data = (image_7[row][col]/255) * R2I_COEF;
            i_valid = 1;
            i_sop = (row == 0) && (col == 0);
            i_eop = (row == IMG_HEIGHT-1) && (col == IMG_WIDTH-1);
        end

        @(posedge clk);
        i_valid = 0;
        i_sop = 0;
        i_eop = 0;


        wait(inst_CNN.o_valid);
        @(posedge clk);
        foreach (image_0[row,col]) begin
            @(posedge clk);
            i_data = (image_2[row][col]/255) * R2I_COEF;
            i_valid = 1;
            i_sop = (row == 0) && (col == 0);
            i_eop = (row == IMG_HEIGHT-1) && (col == IMG_WIDTH-1);
        end

        @(posedge clk);
        i_valid = 0;
        i_sop = 0;
        i_eop = 0;


        wait(inst_CNN.o_valid);
        @(posedge clk);
        foreach (image_0[row,col]) begin
            @(posedge clk);
            i_data = (image_1[row][col]/255) * R2I_COEF;
            i_valid = 1;
            i_sop = (row == 0) && (col == 0);
            i_eop = (row == IMG_HEIGHT-1) && (col == IMG_WIDTH-1);
        end

        @(posedge clk);
        i_valid = 0;
        i_sop = 0;
        i_eop = 0;


        wait(inst_CNN.o_valid);
        @(posedge clk);
        foreach (image_0[row,col]) begin
            @(posedge clk);
            i_data = (image_0[row][col]/255) * R2I_COEF;
            i_valid = 1;
            i_sop = (row == 0) && (col == 0);
            i_eop = (row == IMG_HEIGHT-1) && (col == IMG_WIDTH-1);
        end

        @(posedge clk);
        i_valid = 0;
        i_sop = 0;
        i_eop = 0;
    end



    int first_conv_rows_cntr = 0;
    int first_conv_cols_cntr = 0;
    int first_pool_rows_cntr = 0;
    int first_pool_cols_cntr = 0;

    int second_conv_rows_cntr = 0;
    int second_conv_cols_cntr = 0;
    int second_pool_rows_cntr = 0;
    int second_pool_cols_cntr = 0;

    int flat_cntr = 0;

    real first_conv_data[CONV_DIMENSION[0][1]][IMG_HEIGHT-2][IMG_WIDTH-2];
    real first_relu_data[CONV_DIMENSION[0][1]][IMG_HEIGHT-2][IMG_WIDTH-2];
    real first_pool_data[CONV_DIMENSION[0][1]][(IMG_HEIGHT-2)/2][(IMG_WIDTH-2)/2];

    real second_conv_data[CONV_DIMENSION[1][1]][(IMG_HEIGHT-2)/2-2][(IMG_WIDTH-2)/2-2];
    real second_relu_data[CONV_DIMENSION[1][1]][(IMG_HEIGHT-2)/2-2][(IMG_WIDTH-2)/2-2];
    real second_pool_data[CONV_DIMENSION[1][1]][((IMG_HEIGHT-2)/2-2)/2][((IMG_WIDTH-2)/2-2)/2];

    real flat_data [CONV_DIMENSION[1][1]*(((IMG_HEIGHT-2)/2-2)/2)*(((IMG_HEIGHT-2)/2-2)/2)];


    int first_fc_cntr  = 0;
    int second_fc_cntr = 0;


    real first_fc_data[FLAT_DIMENSION[1]];
    real second_fc_data[CLASSES_QNT];


    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.conv_valid[0])begin
                    foreach (first_conv_data[i]) begin
                        first_conv_data[i][first_conv_rows_cntr][first_conv_cols_cntr] <= $itor($signed(inst_CNN.first_conv_data[i]))/R2I_COEF;
                        first_relu_data[i][first_conv_rows_cntr][first_conv_cols_cntr] <= $itor(inst_CNN.first_relu_data[i])/R2I_COEF;
                    end
                    first_conv_cols_cntr++;
                    if(first_conv_cols_cntr == IMG_WIDTH-2)begin
                        first_conv_cols_cntr = 0;
                        first_conv_rows_cntr++;
                    end
                    if(inst_CNN.conv_eop[0])begin
                        first_conv_cols_cntr = 0;
                        first_conv_rows_cntr = 0;
                    end
                end
    end

    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.pool_valid[0])begin
                    foreach (first_pool_data[i]) begin
                        first_pool_data[i][first_pool_rows_cntr][first_pool_cols_cntr] <= $itor(inst_CNN.first_pool_data[i])/R2I_COEF;
                    end
                    first_pool_cols_cntr++;
                    if(first_pool_cols_cntr == (IMG_HEIGHT-2)/2)begin
                        first_pool_cols_cntr = 0;
                        first_pool_rows_cntr++;
                    end
                    if(inst_CNN.pool_eop[0])begin
                        first_pool_cols_cntr = 0;
                        first_pool_rows_cntr = 0;
                    end
                end
    end

    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.conv_valid[1])begin
                    foreach (second_conv_data[i]) begin
                        second_conv_data[i][second_conv_rows_cntr][second_conv_cols_cntr] <= $itor($signed(inst_CNN.second_conv_data[i]))/R2I_COEF;
                        second_relu_data[i][second_conv_rows_cntr][second_conv_cols_cntr] <= $itor(inst_CNN.second_relu_data[i])/R2I_COEF;
                    end
                    second_conv_cols_cntr++;
                    if(second_conv_cols_cntr == (IMG_HEIGHT-2)/2-2)begin
                        second_conv_cols_cntr = 0;
                        second_conv_rows_cntr++;
                    end
                    if(inst_CNN.conv_eop[1])begin
                        second_conv_cols_cntr = 0;
                        second_conv_rows_cntr = 0;
                    end
                end
    end

    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.pool_valid[1])begin
                    foreach (second_conv_data[i]) begin
                        second_pool_data[i][second_pool_rows_cntr][second_pool_cols_cntr] <= $itor(inst_CNN.second_pool_data[i])/R2I_COEF;
                    end
                    second_pool_cols_cntr++;
                    if(second_pool_cols_cntr == ((IMG_HEIGHT-2)/2-2)/2)begin
                        second_pool_cols_cntr = 0;
                        second_pool_rows_cntr++;
                    end
                    if(inst_CNN.pool_eop[1])begin
                        second_pool_cols_cntr = 0;
                        second_pool_rows_cntr = 0;
                    end
                end
    end

    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.flat_valid)begin
                    flat_data[flat_cntr] <= $itor(inst_CNN.flat_data)/R2I_COEF;
                    flat_cntr++;

                    if(inst_CNN.flat_eop)begin
                        flat_cntr = 0;
                    end
                end
    end

    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.fc_valid[0])begin
                    first_fc_data[first_fc_cntr] <= $itor($signed(inst_CNN.first_fc_relu_data))/R2I_COEF;
                    first_fc_cntr++;

                    if(inst_CNN.fc_eop[0])begin
                        first_fc_cntr = 0;
                    end
                end
    end


    initial begin
        forever
            @(posedge clk)
                if(inst_CNN.fc_valid[1])begin
                    second_fc_data[second_fc_cntr] <= $itor($signed(inst_CNN.fc_data[1]))/R2I_COEF;
                    second_fc_cntr++;

                    if(inst_CNN.fc_eop[1])begin
                        second_fc_cntr = 0;
                    end
                end
    end


    typedef enum {
        ZERO  = 0,
        ONE   = 1,
        TWO   = 2,
        THREE = 3,
        FOUR  = 4,
        FIVE  = 5,
        SIX   = 6,
        SEVEN = 7,
        EIGHT = 8,
        NINE  = 9,

        NONE  = 999
    } e_number;

    e_number detected_class;


    int  detected = 0  ;
    real det_max  = 0.0;
    initial begin
        detected_class = NONE;
            forever wait (
                inst_CNN.fc_eop[1]) begin
                @(posedge clk
            );
                detected_class = ZERO;
                detected       = 0;
                det_max = 0;
                for (int i = 0; i < CLASSES_QNT; i++) begin
                    if(second_fc_data[i] >= det_max)begin
                        detected = i;
                        det_max = second_fc_data[i];
                    end
                end

                for (int i = 0; i < CLASSES_QNT; i++) begin
                    if(detected == i)begin
                        break;
                    end

                    detected_class = detected_class.next();

                end

                @(posedge clk);
                @(posedge clk);

            end
    end




endmodule : CNN_TB



