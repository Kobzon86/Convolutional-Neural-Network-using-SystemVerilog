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

module flat #(
    parameter              PIX_WIDTH  = 8 ,
    parameter              DIMENSION  = 8 ,
    parameter logic [11:0] img_width  = 7,
    parameter logic [11:0] img_height = 7
) (
    input                                       clk    , // Clock
    input                                       clk_en , // Clock Enable
    input                                       rst_n  , // Asynchronous reset active low
    //input pixels
    input        [DIMENSION-1:0][PIX_WIDTH-1:0] i_data ,
    input                                       i_valid,
    input                                       i_sop  ,
    input                                       i_eop  ,
    // output pixels
    output logic [PIX_WIDTH-1:0]                o_data ,
    output logic                                o_valid,
    output logic                                o_sop  ,
    output logic                                o_eop  ,
    ///
    output logic                                o_ready
);


logic [DIMENSION-1:0][img_height-1:0][img_width-1:0][PIX_WIDTH-1:0] img_buf;
wire [DIMENSION*img_height*img_width-1:0][PIX_WIDTH-1:0] img_buf_plain = img_buf;
logic [$clog2(DIMENSION*img_width*img_height)-1:0]o_cntr;

typedef enum logic [2:0] {
    IDLE    = 'd1,
    FILL    = 'd2,
    RELEASE = 'd4
} e_state;

e_state state;


always_ff @(posedge clk) begin
    if(clk_en) begin

        o_valid <= 1'd0;

        o_ready <= 1'b1;
        case (state)
            IDLE: begin
                if (i_valid && i_sop && o_ready) begin
                    state <= FILL;
                end
            end
            FILL: begin
                if(i_valid && i_eop)begin
                    state <= RELEASE;
                    o_cntr <= '0;
                    o_ready <= 1'b0;
                end
            end
            RELEASE: begin

                if(o_cntr == DIMENSION * img_width * img_height)begin
                    state <= IDLE;
                end
                else begin
                    o_valid <= 1'b1;
                    o_data <= img_buf_plain[o_cntr];
                    o_cntr <= o_cntr + 'd1;
                end

                o_sop <= o_cntr == 'd0;
                o_eop <= o_cntr == (DIMENSION * img_width * img_height - 1);
                o_ready <= 1'b0;
            end
            default : state <= IDLE;
        endcase
    end

    if(~rst_n) begin
        o_cntr <= 0;
        state <= IDLE;
    end  
end

always_ff @(posedge clk) begin
    if(clk_en) begin
        if (i_valid && o_ready) begin
            foreach (img_buf[i]) begin
                img_buf[i] <= {i_data[i], img_buf[i][img_height-1:1], img_buf[i][0][img_width-1:1]};
            end
        end
    end
end

endmodule : flat