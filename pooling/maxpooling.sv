// For sim purposes: Image width and height are not changeable dynamically
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


module maxpooling #(
    parameter PIX_WIDTH      = 8 ,
    parameter POOL_DIMENSION = 2 ,
    parameter WIDTH          = 28,
    parameter HEIGHT         = 28
) (
    input                        clk      , // Clock
    input                        clk_en   , // Clock Enable
    input                        rst_n    , // Asynchronous reset active low
    //input pixels
    input        [PIX_WIDTH-1:0] i_data   ,
    input                        i_valid  ,
    input                        i_sop    ,
    input                        i_eop    ,
    // output pixels
    output       [PIX_WIDTH-1:0] o_data   ,
    output                       o_valid  ,
    output                       o_sop    ,
    output                       o_eop    ,
    ///
    output logic                 ready    ,
    output logic [         11:0] cols_cntr,
    output logic [         11:0] rows_cntr
);



/*
    Pixels Delay scheme. if maxpooling 3*3

    pixel_input----------->--------------------                                   --pix[2][2]-->     --pix[2][1]-->
                                /-------\     |                                  |                  |
                         ---<--| FIFO_0 |--<------>----delayed_line[0]------->-----|FF|---------->------|FF|--->-----------pix[2][0]-->--
                         |     \-------/
                         |                                                        --pix[1][2]-->     --pix[1][1]-->
                         |                                                       |                  |
                         ------------------------->---delayed_line[1]------->------|FF|---------->------|FF|--->-----------pix[1][0]-->--
                         |
                         |                                                        --pix[0][2]-->     --pix[0][2]-->
                         |      /-------\                                        |                  |
                         --->--| FIFO_1 |--------->---delayed_line[2]------->------|FF|---------->------|FF|--->-----------pix[0][0]-->--
                               \-------/
*/

    logic [WIDTH-1:0][PIX_WIDTH-1:0] fifo[POOL_DIMENSION-1]; /// !!!!!!!!!!!!!!!!!!!!!!! only for simulation, MUST BE REPLACED by a regular FIFO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    logic [PIX_WIDTH-1:0] delayed_line[POOL_DIMENSION];

    bit [POOL_DIMENSION-2:0][PIX_WIDTH-1:0] after_fifos_ffs[POOL_DIMENSION];

    logic [PIX_WIDTH-1:0] delayed_pix[POOL_DIMENSION][POOL_DIMENSION];

    always_comb begin
        foreach (delayed_line[i]) begin
            delayed_line[i] = (i == 0) ? i_data : fifo[i-1][WIDTH-1];
        end

        foreach (delayed_pix[i,y]) begin
            delayed_pix[i][y] = (y==0)?delayed_line[i] : after_fifos_ffs[i][y-1];
        end
    end

    always_ff @(posedge clk) begin
        if(clk_en && (i_valid || !ready))begin
            foreach (fifo[i]) begin
                fifo[i] <= {fifo[i][WIDTH-2:0],( (i == 0) ? i_data : fifo[i-1][WIDTH-1] )};
            end

            foreach (after_fifos_ffs[i]) begin
                after_fifos_ffs[i] <= {after_fifos_ffs[i],delayed_line[i]};
            end
        end
    end



/*
    maths. if maxpooling 3*3


                 image lines          
     /--      -------------------     --\
     |        | a11 | a12 | a13 |       |
     |        -------------------       |
  max|        | a21 | a22 | a23 |       |   --------> 
     |        -------------------       |
     |        | a31 | a32 | a33 |       |
     \__      -------------------     --/ 
*/
logic [PIX_WIDTH-1:0] max_detected   ;
logic [PIX_WIDTH-1:0] max_detected_ff;

logic [POOL_DIMENSION-1:0][PIX_WIDTH-1:0] max_row_detected   ;
logic [POOL_DIMENSION-1:0][PIX_WIDTH-1:0] max_row_detected_ff;

always_comb begin

    foreach (max_row_detected[i]) begin
        max_row_detected[i] = delayed_pix[i][0];        
    end

    for (int y = 0; y < POOL_DIMENSION; y++) begin
        for (int i = 1; i < POOL_DIMENSION; i++) begin
            if(max_row_detected[y] < delayed_pix[y][i])
                max_row_detected[y] = delayed_pix[y][i];
        end        
    end

    max_detected = max_row_detected_ff[0];
    for (int i = 1; i < POOL_DIMENSION; i++) begin
        if(max_detected < max_row_detected_ff[i])
                max_detected = max_row_detected_ff[i];
    end

end

always_ff @(posedge clk) begin
    if(clk_en) begin
        max_row_detected_ff <= max_row_detected;
        max_detected_ff <= max_detected;
    end
end


/*

*/

    assign o_data = max_detected_ff;

/*
    Latency
*/
    logic [2:0] valid_delay   = '0            ;
    wire        valid_delayed = valid_delay[1];
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            valid_delay <= 0;
        end else begin
            if(clk_en)
                valid_delay <= $size(valid_delay)'( {valid_delay, i_valid && ready} );
        end
    end


/*
    counters
*/

logic [$clog2(POOL_DIMENSION)-1:0]valid_col,valid_row;

    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            cols_cntr <= 0;
            rows_cntr <= 0;
            valid_col   <= 0;
            valid_row   <= 0;
        end else begin
            if(clk_en)begin
                if(valid_delayed)begin
                    cols_cntr <= (cols_cntr == WIDTH-1) ? '0 : (cols_cntr + 'd1);

                    if(valid_col == POOL_DIMENSION-1)
                        valid_col <= '0;
                    else
                        valid_col <= valid_col + 'd1;

                    if(cols_cntr == WIDTH-1)begin
                        rows_cntr <= rows_cntr + 'd1;
                        valid_col   <= '0;
                        valid_row   <= valid_row + 'd1;
                        if (valid_row == POOL_DIMENSION-1) begin
                            valid_row   <= 0;
                        end
                    end
                end
                else if(i_sop)begin
                    cols_cntr <= '0;
                    rows_cntr <= '0;
                    valid_col   <= 0;
                    valid_row   <= 0;
                end
            end
        end
    end

/*
    video control signals
*/
    assign ready = clk_en;


    assign o_valid = valid_delayed && (valid_col == POOL_DIMENSION-1) && (valid_row == POOL_DIMENSION-1) ; 

    assign o_eop = valid_delayed && (valid_col == POOL_DIMENSION-1) && (cols_cntr == WIDTH-(WIDTH[0]+POOL_DIMENSION[0]+1)) && (rows_cntr == HEIGHT-(HEIGHT[0]+POOL_DIMENSION[0]+1));

    assign o_sop = valid_delayed && (valid_col == POOL_DIMENSION-1)  && (rows_cntr == POOL_DIMENSION-1) && (cols_cntr == POOL_DIMENSION-1);





endmodule : maxpooling
