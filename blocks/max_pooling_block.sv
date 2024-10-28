module max_pooling_block #(
    //data width parameters
    parameter PIX_WIDTH      = 8 ,
    //resolution
    parameter IMG_WIDTH      = 28,
    parameter IMG_HEIGHT     = 28,
    parameter POOL_DIMENSION = 2 ,
    //pool_array_parameter
    parameter DIMENSION      = 4
) (
    input                                       clk    ,
    input                                       clk_en ,
    input                                       rst_n  ,
    //input pixels
    input        [DIMENSION-1:0][PIX_WIDTH-1:0] i_data ,
    input                                       i_valid,
    input                                       i_sop  ,
    input                                       i_eop  ,
    // output pixels
    output logic [DIMENSION-1:0][PIX_WIDTH-1:0] o_data ,
    output logic                                o_valid,
    output logic                                o_sop  ,
    output logic                                o_eop  ,
    ///
    output logic                                o_ready
);

    logic valid[DIMENSION];
    logic sop  [DIMENSION];
    logic eop  [DIMENSION];
    logic ready[DIMENSION];

    genvar row,col;

    
    
    generate
        for (row = 0; row < DIMENSION; row++) begin

            maxpooling #(
                .PIX_WIDTH     (PIX_WIDTH     ),
                .POOL_DIMENSION(POOL_DIMENSION),
                .WIDTH         (IMG_WIDTH     ),
                .HEIGHT        (IMG_HEIGHT    )
            ) inst_maxpooling (
                .clk      (clk        ),
                .clk_en   (clk_en     ),
                .rst_n    (rst_n      ),
                .i_data   (i_data[row]),
                .i_valid  (i_valid    ),
                .i_sop    (i_sop      ),
                .i_eop    (i_eop      ),
                .o_data   (o_data[row]),
                .o_valid  (valid[row] ),
                .o_sop    (sop  [row] ),
                .o_eop    (eop  [row] ),
                .ready    (ready[row] ),
                .cols_cntr(           ),
                .rows_cntr(           )
            );


        end
    endgenerate



assign o_valid = valid[0];
assign o_sop = sop[0];
assign o_eop   = eop[0];
assign o_ready = ready[0];




endmodule : max_pooling_block