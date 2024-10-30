// // Convolutional Neural Network  module. For sim purposes:
// // No back preassure in design
// // The math of the module replicates the python script
// //        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
// //        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
// //        self.fc1 = nn.Linear(200, 64)
// //        self.fc2 = nn.Linear(64, 10)
// //
// // To simplify simulation all weights are initialized from "CNN.svh"
// //
// // -----------------------------------------------------------------------------
// // Copyright (c) 2014-2024 All rights reserved
// // -----------------------------------------------------------------------------
// // Author : Maksim Ananev mananev086@gmail.com
// // 
// // Create : 2024-05-13 11:30:23
// // Revise : 2024-10-22 12:20:46
// // Editor : sublime text4, tab size (4)
// // -----------------------------------------------------------------------------


module CNN #(
    parameter                             PIX_WIDTH          = 16     ,
    parameter                             WEIGHT_WIDTH       = 10     ,
    parameter                             FRACT_WIDTH        = 5      ,
    parameter                             CONV_NUMB          = 2      ,
    parameter logic [CONV_NUMB-1:0][1:0][7:0]  CONV_DIMENSION     = {{8'd8, 8'd4}, {8'd4, 8'd1}},   
    parameter logic [CONV_NUMB-1:0][ 3:0] KERNEL_DIMENSION   = {4'd3  , 4'd3},
    parameter                             FLAT_NUMB          = 2      ,
    parameter                             CLASSES_QNT        = 10     ,
    parameter logic [FLAT_NUMB :0][15:0] FLAT_DIMENSION     = {16'd10, 16'd64, 16'd200},
    parameter                             IMG_WIDTH          = 28     ,
    parameter                             IMG_HEIGHT         = 28
) (
    input                                clk                     , // Clock
    input                                clk_en                  , // Clock Enable
    input                                rst_n                   , // Asynchronous reset active low
    //input pixels
    input        [  PIX_WIDTH-1:0]       i_data                  ,
    input                                i_valid                 ,
    input                                i_sop                   ,
    input                                i_eop                   ,
    ///
    output logic                         o_valid                 ,
    output logic [CLASSES_QNT-1:0][31:0] classes                 ,
    ///
    input  int                           weights_mem_in_data     ,
    input  int                           weights_mem_in_addr     ,
    input  int                           weights_mem_sel_addr    ,
    input        [  CONV_NUMB-1:0]       weights_mem_in_kernel_wr,
    input        [  FLAT_NUMB-1:0]       weights_mem_in_fc_wr

);


logic [63:0][PIX_WIDTH-1:0] conv_data[CONV_NUMB];
logic [63:0][PIX_WIDTH-1:0] relu_data[CONV_NUMB];
logic [63:0][PIX_WIDTH-1:0] pool_data[CONV_NUMB];

logic conv_valid[CONV_NUMB];
logic conv_sop  [CONV_NUMB];
logic conv_eop  [CONV_NUMB];

logic pool_valid[CONV_NUMB];
logic pool_sop  [CONV_NUMB];
logic pool_eop  [CONV_NUMB];


genvar numb;
generate
    for (numb = 0; numb < CONV_NUMB; numb++) begin:conv_genloop
        conv_block #(
            .PIX_WIDTH         (PIX_WIDTH                               ),
            .WEIGHT_WIDTH      (WEIGHT_WIDTH                            ),
            .WEIGHT_FRACT_WIDTH(FRACT_WIDTH                             ),
            .TRUNK             ("TRUE"                                  ),
            .IMG_WIDTH         ((IMG_WIDTH - (2**(numb+1)-2))/(2**numb) ),
            .IMG_HEIGHT        ((IMG_HEIGHT - (2**(numb+1)-2))/(2**numb)),
            .KERNEL_DIMENSION  (KERNEL_DIMENSION[numb]                  ),
            .IN_DIMENSION      (CONV_DIMENSION[numb][0]                 ),
            .OUT_DIMENSION     (CONV_DIMENSION[numb][1]                 )
        ) conv_block (
            .clk                     (clk                                       ),
            .clk_en                  (1                                         ),
            .rst_n                   (rst_n                                     ),
            .i_data                  ((numb == 0) ? i_data  : pool_data [numb-1]),
            .i_valid                 ((numb == 0) ? i_valid : pool_valid[numb-1]),
            .i_sop                   ((numb == 0) ? i_sop   : pool_sop  [numb-1]),
            .i_eop                   ((numb == 0) ? i_eop   : pool_eop  [numb-1]),
            .o_data                  (conv_data [numb]                          ),
            .o_valid                 (conv_valid[numb]                          ),
            .o_sop                   (conv_sop  [numb]                          ),
            .o_eop                   (conv_eop  [numb]                          ),
            .weights_mem_in_data     (weights_mem_in_data                       ),
            .weights_mem_in_addr     (weights_mem_in_addr                       ),
            .weights_mem_in_kernel_wr(weights_mem_in_kernel_wr[numb]            ),
            .o_ready                 (                                          )
        );


        relu #(
            .PIX_WIDTH(PIX_WIDTH              ),
            .DIMENSION(CONV_DIMENSION[numb][1])
        ) conv_relu (
            .i_data(conv_data[numb]),
            .o_data(relu_data[numb])
        );


        max_pooling_block #(
            .PIX_WIDTH     (PIX_WIDTH                                 ),
            .IMG_WIDTH     ((IMG_WIDTH - (2**(numb+1)-2))/(2**numb) -2),
            .IMG_HEIGHT    ((IMG_HEIGHT - (2**(numb+1)-2))/(2**numb)-2),
            .POOL_DIMENSION(2                                         ),
            .DIMENSION     (CONV_DIMENSION[numb][1]                   )
        ) max_pooling_block (
            .clk    (clk             ),
            .clk_en (clk_en          ),
            .rst_n  (rst_n           ),
            .i_data (relu_data [numb]),
            .i_valid(conv_valid[numb]),
            .i_sop  (conv_sop  [numb]),
            .i_eop  (conv_eop  [numb]),
            .o_data (pool_data [numb]),
            .o_valid(pool_valid[numb]),
            .o_sop  (pool_sop  [numb]),
            .o_eop  (pool_eop  [numb]),
            .o_ready(                )
        );
    end
    endgenerate







logic [PIX_WIDTH-1:0]flat_data;
logic flat_valid;
logic flat_sop;
logic flat_eop;
logic flat_ready;

    flat #(
        .PIX_WIDTH (PIX_WIDTH                                         ),
        .DIMENSION (CONV_DIMENSION[CONV_NUMB-1][1]                              ),
        .img_width ((IMG_WIDTH - (2**(CONV_NUMB+1)-2))/(2**CONV_NUMB) ),
        .img_height((IMG_HEIGHT - (2**(CONV_NUMB+1)-2))/(2**CONV_NUMB))
    ) inst_flat (
        .clk    (clk                    ),
        .clk_en (clk_en                 ),
        .rst_n  (rst_n                  ),
        .i_data (pool_data [CONV_NUMB-1]),
        .i_valid(pool_valid[CONV_NUMB-1]),
        .i_sop  (pool_sop  [CONV_NUMB-1]),
        .i_eop  (pool_eop  [CONV_NUMB-1]),
        .o_data (flat_data              ),
        .o_valid(flat_valid             ),
        .o_sop  (flat_sop               ),
        .o_eop  (flat_eop               ),
        .o_ready(flat_ready             )
    );

logic [PIX_WIDTH+$clog2(FLAT_DIMENSION[0])-1:0]fc_data[FLAT_NUMB];
logic fc_valid[FLAT_NUMB];
logic fc_sop[FLAT_NUMB];
logic fc_eop[FLAT_NUMB];
logic fc_ready[FLAT_NUMB];

logic [PIX_WIDTH+$clog2(FLAT_DIMENSION[0])-1:0] fc_relu_data[FLAT_NUMB];

generate
    for (numb = 0; numb < FLAT_NUMB; numb++) begin:fc_genloop
        fully_connected_layer #(
            .PIX_WIDTH         (PIX_WIDTH + ((numb == 0) ? 0 : $clog2(FLAT_DIMENSION[0]))),
            .WEIGHT_WIDTH      (WEIGHT_WIDTH                                             ),
            .WEIGHT_FRACT_WIDTH(FRACT_WIDTH                                              ),
            .IN_DIMENSION      (FLAT_DIMENSION[numb]                                     ),
            .OUT_DIMENSION     (FLAT_DIMENSION[numb+1]                                   )
        ) inst_fully_connected_layer1 (
            .clk                 (clk                                           ),
            .clk_en              (clk_en                                        ),
            .rst_n               (rst_n                                         ),
            .i_data              ((numb == 0) ? flat_data : fc_relu_data[numb-1]),
            .i_valid             ((numb == 0) ? flat_valid : fc_valid[numb-1]   ),
            .i_sop               ((numb == 0) ? flat_sop : fc_sop[numb-1]       ),
            .i_eop               ((numb == 0) ? flat_eop : fc_eop[numb-1]       ),
            .o_data              (fc_data[numb]                                 ),
            .o_valid             (fc_valid[numb]                                ),
            .o_sop               (fc_sop[numb]                                  ),
            .o_eop               (fc_eop[numb]                                  ),
            .o_ready             (fc_ready[numb]                                ),
            .weights_mem_in_data (weights_mem_in_data                           ),
            .weights_mem_in_addr (weights_mem_in_addr                           ),
            .weights_mem_sel_addr(weights_mem_sel_addr                          ),
            .weights_mem_in_fc_wr(weights_mem_in_fc_wr[numb]                    )
        );

        relu #(
            .PIX_WIDTH(PIX_WIDTH+$clog2(FLAT_DIMENSION[0])),
            .DIMENSION(1                )
        ) fc_relu (
            .i_data(fc_data[numb]     ),
            .o_data(fc_relu_data[numb])
        );
    end
endgenerate






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

