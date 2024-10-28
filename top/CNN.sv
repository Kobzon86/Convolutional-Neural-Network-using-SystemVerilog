// Convolutional Neural Network  module. For sim purposes: Not fully parameterizable
// The math of the module replicates the python script
//        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
//        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
//        self.fc1 = nn.Linear(200, 64)
//        self.fc2 = nn.Linear(64, 10)
//
// To simplify simulation all weights are initialized from "CNN.svh"
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



module CNN #(
    parameter                             PIX_WIDTH          = 16     ,
    parameter                             WEIGHT_WIDTH       = 10     ,
    parameter                             FRACT_WIDTH        = 5      ,
    parameter                             CONV_NUMB          = 2      ,
    parameter logic [CONV_NUMB-1:0][1:0][7:0]  CONV_DIMENSION     = {{8'd8, 8'd4}, {8'd4, 8'd1}},   
    parameter logic [CONV_NUMB-1:0][ 3:0] KERNEL_DIMENSION   = {4'd3  , 4'd3},
    parameter                             FLAT_NUMB          = 2      ,
    parameter logic [FLAT_NUMB-1:0][15:0] FLAT_DIMENSION     = {16'd64, 16'd200},
    parameter                             CLASSES_QNT        = 10     ,
    parameter                             IMG_WIDTH          = 28     ,
    parameter                             IMG_HEIGHT         = 28
) (
    input                      clk    , // Clock
    input                      clk_en , // Clock Enable
    input                      rst_n  , // Asynchronous reset active low
    //input pixels
    input      [PIX_WIDTH-1:0] i_data ,
    input                      i_valid,
    input                      i_sop  ,
    input                      i_eop  ,
    ///

    output logic                    o_valid,
    output logic [CLASSES_QNT-1:0][31:0] classes 
);




localparam R2I_COEF = 2**FRACT_WIDTH;

`include "CNN.svh"

logic [CONV_DIMENSION[0][1]-1:0][CONV_DIMENSION[0][0]-1:0][KERNEL_DIMENSION[0]-1:0][KERNEL_DIMENSION[0]-1:0][WEIGHT_WIDTH-1:0] kernel_1;
logic [CONV_DIMENSION[1][1]-1:0][CONV_DIMENSION[1][0]-1:0][KERNEL_DIMENSION[0]-1:0][KERNEL_DIMENSION[0]-1:0][WEIGHT_WIDTH-1:0] kernel_2;

logic [CONV_DIMENSION[0][1]-1:0][WEIGHT_WIDTH-1:0]bias_1;
logic [CONV_DIMENSION[1][1]-1:0][WEIGHT_WIDTH-1:0]bias_2;


initial begin
    foreach (kernel_1[dim2, dim1, row, col]) begin
        kernel_1[dim2][dim1][row][col] = R2I_COEF*kernel_1_re[dim2][dim1][row][col];
    end
    foreach (kernel_2[dim2, dim1, row, col]) begin
        kernel_2[dim2][dim1][row][col] = R2I_COEF*kernel_2_re[dim2][dim1][row][col];
    end
    foreach (bias_1[dim1]) begin
        bias_1[dim1] = R2I_COEF*conv_1_bias_re[dim1];
    end
    foreach (bias_2[dim1]) begin
        bias_2[dim1] = R2I_COEF*conv_2_bias_re[dim1];
    end

end

logic [CONV_DIMENSION[0][1]-1:0][PIX_WIDTH-1:0] first_conv_data;
logic [CONV_DIMENSION[0][1]-1:0][PIX_WIDTH-1:0] first_relu_data;
logic [CONV_DIMENSION[0][1]-1:0][PIX_WIDTH-1:0] first_pool_data;

logic [CONV_DIMENSION[1][1]-1:0][PIX_WIDTH-1:0] second_conv_data;
logic [CONV_DIMENSION[1][1]-1:0][PIX_WIDTH-1:0] second_relu_data;
logic [CONV_DIMENSION[1][1]-1:0][PIX_WIDTH-1:0] second_pool_data;

logic conv_valid[CONV_NUMB];
logic conv_sop  [CONV_NUMB];
logic conv_eop  [CONV_NUMB];

logic pool_valid[CONV_NUMB];
logic pool_sop  [CONV_NUMB];
logic pool_eop  [CONV_NUMB];



conv_block #(
    .PIX_WIDTH         (PIX_WIDTH           ),
    .WEIGHT_WIDTH      (WEIGHT_WIDTH        ),
    .WEIGHT_FRACT_WIDTH(FRACT_WIDTH         ),
    .TRUNK             ("TRUE"              ),
    .IMG_WIDTH         (IMG_WIDTH           ),
    .IMG_HEIGHT        (IMG_HEIGHT          ),
    .KERNEL_DIMENSION  (KERNEL_DIMENSION[0] ),
    .IN_DIMENSION      (CONV_DIMENSION[0][0]),
    .OUT_DIMENSION     (CONV_DIMENSION[0][1])
) first_conv_block (
    .clk    (clk            ),
    .clk_en (1              ),
    .rst_n  (rst_n          ),
    .i_data (i_data         ),
    .i_valid(i_valid        ),
    .i_sop  (i_sop          ),
    .i_eop  (i_eop          ),
    .o_data (first_conv_data),
    .o_valid(conv_valid[0]  ),
    .o_sop  (conv_sop  [0]  ),
    .o_eop  (conv_eop  [0]  ),
    .kernel (kernel_1       ),
    .bias   (bias_1         ),
    .o_ready(               )
);

relu #(
    .PIX_WIDTH(PIX_WIDTH           ),
    .DIMENSION(CONV_DIMENSION[0][1])
) first_relu (
    .i_data(first_conv_data),
    .o_data(first_relu_data)
);

max_pooling_block #(
    .PIX_WIDTH     (PIX_WIDTH           ),
    .IMG_WIDTH     (IMG_WIDTH-2         ),
    .IMG_HEIGHT    (IMG_HEIGHT-2        ),
    .POOL_DIMENSION(2                   ),
    .DIMENSION     (CONV_DIMENSION[0][1])
) first_max_pooling_block (
    .clk    (clk            ),
    .clk_en (clk_en         ),
    .rst_n  (rst_n          ),
    .i_data (first_relu_data),
    .i_valid(conv_valid[0]  ),
    .i_sop  (conv_sop  [0]  ),
    .i_eop  (conv_eop  [0]  ),
    .o_data (first_pool_data),
    .o_valid(pool_valid[0]  ),
    .o_sop  (pool_sop  [0]  ),
    .o_eop  (pool_eop  [0]  ),
    .o_ready(               )
);



conv_block #(
    .PIX_WIDTH         (PIX_WIDTH           ),
    .WEIGHT_WIDTH      (WEIGHT_WIDTH        ),
    .WEIGHT_FRACT_WIDTH(FRACT_WIDTH         ),
    .TRUNK             ("TRUE"              ),
    .IMG_WIDTH         ((IMG_WIDTH-2)/2     ),
    .IMG_HEIGHT        ((IMG_HEIGHT-2)/2    ),
    .KERNEL_DIMENSION  (KERNEL_DIMENSION[0] ),
    .IN_DIMENSION      (CONV_DIMENSION[1][0]),
    .OUT_DIMENSION     (CONV_DIMENSION[1][1])
) sec_conv_block (
    .clk    (clk             ),
    .clk_en (1               ),
    .rst_n  (rst_n           ),
    .i_data (first_pool_data ),
    .i_valid(pool_valid[0]   ),
    .i_sop  (pool_sop  [0]   ),
    .i_eop  (pool_eop  [0]   ),
    .o_data (second_conv_data),
    .o_valid(conv_valid[1]   ),
    .o_sop  (conv_sop  [1]   ),
    .o_eop  (conv_eop  [1]   ),
    .kernel (kernel_2        ),
    .bias   (bias_2          ),
    .o_ready(                )
);

relu #(
    .PIX_WIDTH(PIX_WIDTH           ),
    .DIMENSION(CONV_DIMENSION[1][1])
) sec_relu (
    .i_data(second_conv_data),
    .o_data(second_relu_data)
);

max_pooling_block #(
    .PIX_WIDTH     (PIX_WIDTH           ),
    .IMG_WIDTH     ((IMG_WIDTH-2)/2 -2  ),
    .IMG_HEIGHT    ((IMG_HEIGHT-2)/2-2  ),
    .POOL_DIMENSION(2                   ),
    .DIMENSION     (CONV_DIMENSION[1][1])
) sec_max_pooling_block (
    .clk    (clk             ),
    .clk_en (clk_en          ),
    .rst_n  (rst_n           ),
    .i_data (second_relu_data),
    .i_valid(conv_valid[1]   ),
    .i_sop  (conv_sop  [1]   ),
    .i_eop  (conv_eop  [1]   ),
    .o_data (second_pool_data),
    .o_valid(pool_valid[1]   ),
    .o_sop  (pool_sop  [1]   ),
    .o_eop  (pool_eop  [1]   ),
    .o_ready(                )
);


logic [PIX_WIDTH-1:0]flat_data;
logic flat_valid;
logic flat_sop;
logic flat_eop;
logic flat_ready;

    flat #(
        .PIX_WIDTH (PIX_WIDTH             ),
        .DIMENSION (CONV_DIMENSION[1][1]  ),
        .img_width (((IMG_WIDTH-2)/2 -2)/2),
        .img_height(((IMG_HEIGHT-2)/2-2)/2)
    ) inst_flat (
        .clk    (clk             ),
        .clk_en (clk_en          ),
        .rst_n  (rst_n           ),
        .i_data (second_pool_data),
        .i_valid(pool_valid[1]   ),
        .i_sop  (pool_sop[1]     ),
        .i_eop  (pool_eop[1]     ),
        .o_data (flat_data       ),
        .o_valid(flat_valid      ),
        .o_sop  (flat_sop        ),
        .o_eop  (flat_eop        ),
        .o_ready(flat_ready      )
    );


logic [PIX_WIDTH+$clog2(CONV_DIMENSION[1][1])-1:0]fc_data[2];
logic fc_valid[2];
logic fc_sop[2];
logic fc_eop[2];
logic fc_ready[2];


    fully_connected_layer #(
        .PIX_WIDTH         (PIX_WIDTH        ),
        .WEIGHT_WIDTH      (WEIGHT_WIDTH     ),
        .WEIGHT_FRACT_WIDTH(FRACT_WIDTH      ),
        .IN_DIMENSION      (FLAT_DIMENSION[0]),
        .OUT_DIMENSION     (FLAT_DIMENSION[1])
    ) inst_fully_connected_layer1 (
        .clk    (clk        ),
        .clk_en (clk_en     ),
        .rst_n  (rst_n      ),
        .i_data (flat_data  ),
        .i_valid(flat_valid ),
        .i_sop  (flat_sop   ),
        .i_eop  (flat_eop   ),
        .o_data (fc_data[0] ),
        .o_valid(fc_valid[0]),
        .o_sop  (fc_sop[0]  ),
        .o_eop  (fc_eop[0]  ),
        .o_ready(fc_ready[0])
    );


logic [PIX_WIDTH+$clog2(CONV_DIMENSION[1][1])-1:0] first_fc_relu_data;


relu #(
    .PIX_WIDTH(PIX_WIDTH+$clog2(CONV_DIMENSION[1][1])),
    .DIMENSION(CONV_DIMENSION[1][1]                  )
) fc_first_relu (
    .i_data(fc_data[0]        ),
    .o_data(first_fc_relu_data)
);



    fully_connected_layer #(
        .PIX_WIDTH         (PIX_WIDTH+$clog2(CONV_DIMENSION[1][1])),
        .WEIGHT_WIDTH      (WEIGHT_WIDTH                          ),
        .WEIGHT_FRACT_WIDTH(FRACT_WIDTH                           ),
        .IN_DIMENSION      (FLAT_DIMENSION[1]                     ),
        .OUT_DIMENSION     (CLASSES_QNT                           )
    ) inst_fully_connected_layer2 (
        .clk    (clk               ),
        .clk_en (clk_en            ),
        .rst_n  (rst_n             ),
        .i_data (first_fc_relu_data),
        .i_valid(fc_valid[0]       ),
        .i_sop  (fc_sop[0]         ),
        .i_eop  (fc_eop[0]         ),
        .o_data (fc_data[1]        ),
        .o_valid(fc_valid[1]       ),
        .o_sop  (fc_sop[1]         ),
        .o_eop  (fc_eop[1]         ),
        .o_ready(fc_ready[1]       )
    );


int classes_cntr = 0;
always_ff @(posedge clk or negedge rst_n) begin
    if(clk_en)begin
        
        o_valid <= 'd0;

        if(fc_valid[FLAT_NUMB-1])begin

            classes[classes_cntr] <= $signed(fc_data[FLAT_NUMB-1]);

            if(fc_eop[FLAT_NUMB-1])begin
                classes_cntr <= '0;
                o_valid <= 'd1;
            end
            else 
                classes_cntr <= classes_cntr + 'd1;

        end

    end


    if(~rst_n) begin
        classes <= '0;
        classes_cntr <= '0;
    end
end

endmodule : CNN