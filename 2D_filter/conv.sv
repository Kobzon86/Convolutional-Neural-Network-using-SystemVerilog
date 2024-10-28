// 2D Convolution module. For sim purposes: Image width and height are not changeable dynamically
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



// `define RELU

module conv #(
    parameter PIX_WIDTH          = 8  ,
    parameter WEIGHT_WIDTH       = 10 ,
    parameter WEIGHT_FRACT_WIDTH = 5  ,
    parameter KERNEL_DIMENSION   = 3  ,
    parameter TRUNK = "TRUE",
    parameter logic [         11:0] img_width  = 28,
    parameter logic [         11:0] img_height = 28
    
) (
    input                        clk       , // Clock
    input                        clk_en    , // Clock Enable
    input                        rst_n     , // Asynchronous reset active low
    //input pixels
    input        [PIX_WIDTH-1:0] i_data    ,
    input                        i_valid   ,
    input                        i_sop     ,
    input                        i_eop     ,
    // output pixels
    output       [((TRUNK == "TRUE") ? PIX_WIDTH : (PIX_WIDTH+WEIGHT_FRACT_WIDTH))-1:0] o_data    ,
    output                       o_valid   ,
    output                       o_sop     ,
    output                       o_eop     ,
    ///
    input [KERNEL_DIMENSION-1:0][KERNEL_DIMENSION-1:0][WEIGHT_WIDTH-1:0] kernel    ,
    // input        [         11:0] img_width ,
    // input        [         11:0] img_height,
    output logic                 ready     ,
    output logic [         11:0] cols_cntr ,
    output logic [         11:0] rows_cntr
);



/*
    Pixels Delay scheme. if Kernel 3*3

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
 localparam MAX_DEPTH = 1920;

    logic [img_width-1:0][PIX_WIDTH-1:0] fifo[KERNEL_DIMENSION-1]; /// !!!!!!!!!!!!!!!!!!!!!!! only for simulation, MUST BE REPLACED by a regular FIFO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    logic [PIX_WIDTH-1:0] delayed_line[KERNEL_DIMENSION];

    bit [KERNEL_DIMENSION-2:0][PIX_WIDTH-1:0] after_fifos_ffs[KERNEL_DIMENSION];

    logic [PIX_WIDTH-1:0] delayed_pix[KERNEL_DIMENSION][KERNEL_DIMENSION];

    always_comb begin
        foreach (delayed_line[i]) begin
            delayed_line[i] = (i == 0) ? i_data : fifo[i-1][img_width-1];
        end

        foreach (delayed_pix[i,y]) begin
            delayed_pix[i][y] = (y==0)?delayed_line[i] : after_fifos_ffs[i][y-1];
        end
    end

    always_ff @(posedge clk) begin
        if(clk_en && (i_valid || !ready))begin
            foreach (fifo[i]) begin
                fifo[i] <= {fifo[i][img_width-2:0],( (i == 0) ? i_data : fifo[i-1][img_width-1] )};
            end

            foreach (after_fifos_ffs[i]) begin
                after_fifos_ffs[i] <= {after_fifos_ffs[i],delayed_line[i]};
            end
        end
    end



/*
    Convolution's maths. if Kernel 3*3


    kernel table                     image lines                                          multiplication table
    -------------------       -----------------------------------------      -----------------------------------------------------
    | a11 | a12 | a13 |       | pix[i][y] | pix[i][y+1] | pix[i][y+2] |      | a11*pix[i][y+2] | a12*pix[i][y+1] | a13*pix[i][y] |
    -------------------       -----------------------------------------      -----------------------------------------------------
    | a21 | a22 | a23 |   X   | pix[i][y] | pix[i][y+1] | pix[i][y+2] |   =  | a21*pix[i][y+2] | a22*pix[i][y+1] | a23*pix[i][y] |   --------> Sum(multiplication table)
    -------------------       -----------------------------------------      -----------------------------------------------------
    | a31 | a32 | a33 |       | pix[i][y] | pix[i][y+1] | pix[i][y+2] |      | a31*pix[i][y+2] | a32*pix[i][y+1] | a33*pix[i][y] |
    -------------------       -----------------------------------------      -----------------------------------------------------
*/
    logic signed [KERNEL_DIMENSION-1:0][KERNEL_DIMENSION-1:0][WEIGHT_WIDTH+PIX_WIDTH-1:0] mult_result;

    always_ff @(posedge clk) begin : proc_multiplying
        if(clk_en)begin
            foreach (mult_result[i,y]) begin
                mult_result[i][y] <= $signed({1'b0, delayed_pix[(KERNEL_DIMENSION-1)-i][(KERNEL_DIMENSION-1)-y]}) * $signed(kernel[i][y]);
            end
        end
    end

    logic signed [$clog2(KERNEL_DIMENSION)+$size(mult_result,3)-1:0]mult_intermed_sum_1dim[KERNEL_DIMENSION];
    logic signed [$clog2(KERNEL_DIMENSION)+$size(mult_result,3)-1:0]mult_sum_1dim[KERNEL_DIMENSION];
    logic signed [$clog2(KERNEL_DIMENSION)+$size(mult_sum_1dim,2)-1:0]mult_intermed_sum_2dim;

    always_comb begin
        mult_intermed_sum_2dim = '0;
        foreach (mult_intermed_sum_1dim[i]) begin
            mult_intermed_sum_1dim[i] = '0;
            foreach (mult_intermed_sum_1dim[y]) begin
                mult_intermed_sum_1dim[i] += $signed(mult_result[i][y]);
            end

            mult_intermed_sum_2dim += mult_sum_1dim[i];

        end
    end


    logic signed [$clog2(KERNEL_DIMENSION)+$size(mult_sum_1dim,2)-1:0]mult_sum_out;
    always_ff @(posedge clk) begin : proc_mult_sum
        if(clk_en)begin
            foreach (mult_sum_1dim[i]) begin
                mult_sum_1dim[i] <= mult_intermed_sum_1dim[i];
            end            
            mult_sum_out <= (TRUNK == "TRUE") ? (mult_intermed_sum_2dim>>>WEIGHT_FRACT_WIDTH) : mult_intermed_sum_2dim;
        end

    end


/*
    normalize( Sum(multiplication table) ) -----> pixel output

    normalize(a, min = 0, max = 255){
        if(a < min)
            a = 0;
        if(a > max)
            a = max;

    }
*/
`ifdef RELU
    assign o_data = ( mult_sum_out < 0 ) ? '0 : ( ( |mult_sum_out[$size( mult_sum_out )-1:PIX_WIDTH] ) ? ( 2**PIX_WIDTH - 1 ) : mult_sum_out);
`else 
    assign o_data = mult_sum_out;
`endif
/*
    Latency
*/
    logic [2:0] valid_delay   = '0            ;
    wire        valid_delayed = valid_delay[2];
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            valid_delay <= 0;
        end else begin
            if(clk_en)
                valid_delay <= {valid_delay, i_valid && ready};
        end
    end


/*
    counters
*/
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            cols_cntr <= 0;
            rows_cntr <= 0;
        end else begin
            if(clk_en)begin
                if(valid_delayed || (!ready && (rows_cntr == img_height)))begin
                    cols_cntr <= (cols_cntr == img_width-1) ? '0 : (cols_cntr + 'd1);
                    if(cols_cntr == img_width-1)
                        rows_cntr <= (rows_cntr == img_height) ? '0 : (rows_cntr + 'd1);
                end
                else if(i_sop)begin
                    cols_cntr <= '0;
                    rows_cntr <= '0;
                end
            end
        end
    end

/*
    video control signals
*/
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n) begin
            ready <= 1;
        end else if(clk_en) begin
            if (i_eop) begin
                ready <= 1'b0;
            end
            else if(rows_cntr == 0)
                ready <= 1'b1;
        end
    end


    assign o_valid = valid_delayed && (rows_cntr > 1) && (rows_cntr < img_height) && (cols_cntr > 1) && (cols_cntr < img_width);

    assign o_eop = valid_delayed && (cols_cntr == img_width-1) && (rows_cntr == img_height-1);

    assign o_sop = valid_delayed && (rows_cntr == 2) && (cols_cntr == 2);





endmodule : conv
