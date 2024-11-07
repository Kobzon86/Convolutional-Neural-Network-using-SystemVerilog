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
    parameter PIX_WIDTH          = 16  ,
    parameter WEIGHT_WIDTH       = 16 ,
    parameter WEIGHT_FRACT_WIDTH = 10  ,
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
    input  [$clog2(IN_DIMENSION)-1:0]                 weights_mem_in_addr ,
    input  [$clog2(OUT_DIMENSION):0]                  weights_mem_sel_addr,
    input                                             weights_mem_in_fc_wr,
    ///
    output logic                                      o_ready
);




int col_cntr;
logic [OUT_DIMENSION :0] weight_wr;

always_comb begin 
   weight_wr = '0;
   weight_wr[weights_mem_sel_addr] = weights_mem_in_fc_wr;
end

genvar y;

logic [WEIGHT_WIDTH-1:0]weights[OUT_DIMENSION];

generate 
for(y = 0; y < OUT_DIMENSION; y++)begin
    single_port_rom #(
    .ADDR_WIDTH($clog2(IN_DIMENSION)),
    .DATA_WIDTH(WEIGHT_WIDTH)
    )
    weight_rom(
     .clk(clk),
     .w_addr(weights_mem_in_addr),
     .r_addr(col_cntr),
     .data(weights_mem_in_data),
     .o(weights[y]),
     .we(weight_wr[y])
    
    );
end

endgenerate


       

logic [WEIGHT_WIDTH-1:0]bias[OUT_DIMENSION];

logic [PIX_WIDTH-1:0] i_data_ff;
logic i_sop_ff;
logic o_ready_ff;
logic i_valid_ff;
always_ff @(posedge clk) begin

    if(weight_wr[OUT_DIMENSION])
        bias[weights_mem_in_addr] <= weights_mem_in_data;

    i_data_ff <= i_data;
    i_valid_ff <= i_valid;
    i_sop_ff <= i_sop;
    o_ready_ff <= o_ready;

    if(~rst_n) begin
        i_data_ff <= '0;
        i_sop_ff <= '0;
        o_ready_ff <= '0;
        i_valid_ff <= '0;
    end
end



logic [PIX_WIDTH*WEIGHT_FRACT_WIDTH+$clog2(OUT_DIMENSION)-1:0] integrators[OUT_DIMENSION];


typedef enum logic [2:0] {
    IDLE    = 'd1,
    FILL    = 'd2,
    RELEASE = 'd4
} e_state;

e_state state;

logic fill_delay;

always_ff @(posedge clk) begin
    if(clk_en) begin

        o_valid    <= 1'd0;
        o_ready    <= 1'b1;
        fill_delay <= 1'b0;

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
                        fill_delay <= 1'b1;
                    end
                end
                
            end
            RELEASE: begin

                if(col_cntr == OUT_DIMENSION)begin
                    state <= IDLE;
                end
                else begin
                    o_data <= $signed(integrators[col_cntr])/(2**WEIGHT_FRACT_WIDTH) + $signed(bias[col_cntr]);
                    o_valid <= !fill_delay;
                    col_cntr <= col_cntr + $size(col_cntr)'(!fill_delay);
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
        if(i_valid_ff && o_ready_ff)

            foreach (integrators[x]) begin
                if(i_sop_ff)
                    integrators[x] <= $signed(weights[x])*$signed(i_data_ff);
                else
                    integrators[x] <= $signed(weights[x])*$signed(i_data_ff) + $signed(integrators[x]);
            end
    end
end


endmodule : fully_connected_layer


module single_port_rom
  # (parameter ADDR_WIDTH = 4,
     parameter DATA_WIDTH = 32
    )

  ( 	input 					clk,
   		input [ADDR_WIDTH-1:0]	r_addr,
   		input [ADDR_WIDTH-1:0]	w_addr,
   		input  [DATA_WIDTH-1:0]	data,
   		output logic [DATA_WIDTH-1:0]	o,
   		input 					we
  );

  reg [DATA_WIDTH-1:0] 	mem [2**ADDR_WIDTH];

  always @ (posedge clk) begin
    if (we)
      mem[w_addr] <= data;
      
    o <=  mem[r_addr];
  end

    
endmodule
