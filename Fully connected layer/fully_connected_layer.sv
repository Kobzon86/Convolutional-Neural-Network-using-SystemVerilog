// Fully connected layer of CNN. For sim purposes: 
// Image width and height are not changeable dynamically
// To simplify simulation all weights initialize from "CNN.svh"  -- Temporal
// -----------------------------------------------------------------------------
// Copyright (c) 2014-2024 All rights reserved
// -----------------------------------------------------------------------------
// Author : Maksim Ananev mananev086@gmail.com
// 
// Create : 2024-05-13 11:30:23
// Revise : 2024-10-22 12:20:46
// Editor : sublime text4, tab size (4)
// -----------------------------------------------------------------------------

module fully_connected_layer #(
    //data width parameters
    parameter PIX_WIDTH          = 8  ,
    parameter WEIGHT_WIDTH       = 10 ,
    parameter WEIGHT_FRACT_WIDTH = 5  ,
    //array_parameter
    parameter IN_DIMENSION       = 200,
    parameter OUT_DIMENSION      = 64
) (
    input                                             clk                 , // Clock
    input                                             clk_en              , // Clock Enable
    input                                             rst_n               , // Asynchronous reset active low
    //input pixels
    input        [                     PIX_WIDTH-1:0] i_data              ,
    input                                             i_valid             ,
    input                                             i_sop               ,
    input                                             i_eop               ,
    // output pixels
    output logic [PIX_WIDTH+$clog2(IN_DIMENSION)-1:0] o_data              ,
    output logic                                      o_valid             ,
    output logic                                      o_sop               ,
    output logic                                      o_eop               ,
    ///
    input  int                                        weights_mem_in_data ,
    input  int                                        weights_mem_in_addr ,
    input  int                                        weights_mem_sel_addr,
    input                                             weights_mem_in_fc_wr,
    ///
    output logic                                      o_ready
);

logic [WEIGHT_WIDTH-1:0]weights[OUT_DIMENSION][IN_DIMENSION];
logic [WEIGHT_WIDTH-1:0]bias[OUT_DIMENSION];


always_ff @(posedge clk) begin
    if(weights_mem_in_fc_wr)
        if(weights_mem_sel_addr == OUT_DIMENSION)
            bias[weights_mem_in_addr] <= weights_mem_in_data;
        else
            weights[weights_mem_sel_addr][weights_mem_in_addr] <= weights_mem_in_data;

    if(~rst_n) begin
    end
end

int col_cntr;

logic [PIX_WIDTH*WEIGHT_FRACT_WIDTH+$clog2(OUT_DIMENSION)-1:0] integrators[OUT_DIMENSION];


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
                
                col_cntr <= '0;

                if (i_valid && i_sop && o_ready) begin
                    state <= FILL;
                    col_cntr <= col_cntr + 'd1;
                end
            end
            FILL: begin
                if (i_valid) begin
                    col_cntr <= col_cntr + 'd1;

                    if(i_eop)begin
                        state <= RELEASE;
                        col_cntr <= '0;
                        o_ready <= 1'b0;
                    end
                end
                
            end
            RELEASE: begin

                if(col_cntr == OUT_DIMENSION)begin
                    state <= IDLE;
                end
                else begin
                    o_valid <= 1'b1;
                    o_data <= $signed(integrators[col_cntr])/(2**WEIGHT_FRACT_WIDTH) + $signed(bias[col_cntr]);
                    col_cntr <= col_cntr + 'd1;
                end

                o_sop <= col_cntr == 'd0;
                o_eop <= col_cntr == (OUT_DIMENSION- 1);
                o_ready <= 1'b0;

            end

            default : state <= IDLE;
        endcase
    end

    if(~rst_n) begin
        col_cntr <= 0;
        state <= IDLE;
    end 
end

always_ff @(posedge clk) begin
    if(clk_en) begin
        if(i_valid && o_ready)

            foreach (integrators[x]) begin
                if(i_sop)
                    integrators[x] <= $signed(weights[x][col_cntr])*$signed(i_data);
                else
                    integrators[x] <= $signed(weights[x][col_cntr])*$signed(i_data) + $signed(integrators[x]);
            end
    end
end







endmodule : fully_connected_layer