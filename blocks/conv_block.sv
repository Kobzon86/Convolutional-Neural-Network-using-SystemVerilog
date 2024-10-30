
module conv_block #(
    //data width parameters
    parameter PIX_WIDTH          = 8     ,
    parameter WEIGHT_WIDTH       = 10    ,
    parameter WEIGHT_FRACT_WIDTH = 5     ,
    parameter TRUNK              = "TRUE",
    //resolution
    parameter IMG_WIDTH          = 28    ,
    parameter IMG_HEIGHT         = 28    ,
    //conv_array_parameter
    parameter KERNEL_DIMENSION   = 3     ,
    parameter IN_DIMENSION       = 1     ,
    parameter OUT_DIMENSION      = 4
) (
    input                                              clk                              ,
    input                                              clk_en                           ,
    input                                              rst_n                            ,
    //input pixels
    input        [ IN_DIMENSION-1:0][   PIX_WIDTH-1:0] i_data                           ,
    input                                              i_valid                          ,
    input                                              i_sop                            ,
    input                                              i_eop                            ,
    // output pixels
    output                                             logic [OUT_DIMENSION-1:0][((TRUNK  == "TRUE") ? PIX_WIDTH : (PIX_WIDTH+WEIGHT_FRACT_WIDTH))-1:0] o_data ,
    output logic                                       o_valid                          ,
    output logic                                       o_sop                            ,
    output logic                                       o_eop                            ,
    ///
    input  int                                         weights_mem_in_data              ,
    input  int                                         weights_mem_in_addr              ,
    input                                              weights_mem_in_kernel_wr         ,
    ///
    // input        [OUT_DIMENSION-1:0][WEIGHT_WIDTH-1:0] bias                             ,
    ///
    output logic                                       o_ready
);


    logic [OUT_DIMENSION + OUT_DIMENSION * IN_DIMENSION * KERNEL_DIMENSION * KERNEL_DIMENSION - 1 : 0][WEIGHT_WIDTH-1:0] kernel_plain;
    wire [OUT_DIMENSION-1:0][IN_DIMENSION-1:0][KERNEL_DIMENSION-1:0][KERNEL_DIMENSION-1:0][WEIGHT_WIDTH-1:0] kernel;
    wire        [OUT_DIMENSION-1:0][WEIGHT_WIDTH-1:0] bias;
    assign {bias,kernel} = kernel_plain;

always_ff @(posedge clk) begin 
    if(weights_mem_in_kernel_wr) 
       kernel_plain[weights_mem_in_addr] <= weights_mem_in_data;  

    if(~rst_n) begin
    end
end

    logic signed [((TRUNK == "TRUE") ? PIX_WIDTH : (PIX_WIDTH+WEIGHT_FRACT_WIDTH))-1:0] conv_outputs[OUT_DIMENSION][IN_DIMENSION];

    logic valid[OUT_DIMENSION][IN_DIMENSION];
    logic sop  [OUT_DIMENSION][IN_DIMENSION];
    logic eop  [OUT_DIMENSION][IN_DIMENSION];
    logic ready[OUT_DIMENSION][IN_DIMENSION];
    
    
    genvar row,col;
    generate
        for (row = 0; row < OUT_DIMENSION; row++) begin
            for (col = 0; col < IN_DIMENSION; col++) begin

                conv #(
                    .PIX_WIDTH         (PIX_WIDTH         ),
                    .WEIGHT_WIDTH      (WEIGHT_WIDTH      ),
                    .WEIGHT_FRACT_WIDTH(WEIGHT_FRACT_WIDTH),
                    .TRUNK             (TRUNK             ),
                    .KERNEL_DIMENSION  (KERNEL_DIMENSION  ),
                    .img_width         (IMG_WIDTH         ),
                    .img_height        (IMG_HEIGHT        )
                ) inst_conv (
                    .clk      (clk                   ),
                    .clk_en   (clk_en                ),
                    .rst_n    (rst_n                 ),
                    .i_data   (i_data[col]           ),
                    .i_valid  (i_valid               ),
                    .i_sop    (i_sop                 ),
                    .i_eop    (i_eop                 ),
                    .o_data   (conv_outputs[row][col]),
                    .o_valid  (valid[row][col]       ),
                    .o_sop    (sop  [row][col]       ),
                    .o_eop    (eop  [row][col]       ),
                    .kernel   (kernel[row][col]      ),
                    .ready    (ready[row][col]       ),
                    .cols_cntr(                      ),
                    .rows_cntr(                      )
                );

            end
        end
    endgenerate


    logic [((TRUNK == "TRUE") ? PIX_WIDTH : (PIX_WIDTH+WEIGHT_FRACT_WIDTH))-1:0] sum[OUT_DIMENSION];

    always_comb begin

        foreach (sum[x]) begin
            sum[x] = '0;
        end

        foreach (conv_outputs[x,z]) begin
            sum[x] += $signed(conv_outputs[x][z]);
        end
    end


    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            o_valid <= 0;
            o_sop   <= 0;
            o_eop   <= 0;
        end else if(clk_en) begin
            o_valid <= valid[0][0];
            o_sop   <= sop[0][0];
            o_eop   <= eop[0][0];

            foreach (o_data[x]) begin
                o_data[x] <= $signed(sum[x]) + $signed(bias[x]);
            end


        end
    end

assign o_ready = ready[0][0];


endmodule : conv_block