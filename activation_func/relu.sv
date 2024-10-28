module relu #(
	parameter PIX_WIDTH = 8,
	parameter DIMENSION = 4
) (
	//input pixels
	input        [DIMENSION-1:0][PIX_WIDTH-1:0] i_data ,
	// output pixels
	output logic [DIMENSION-1:0][PIX_WIDTH-1:0] o_data 
);


always_comb begin
	foreach (o_data[i]) begin
		o_data[i] = i_data[i][PIX_WIDTH-1] ? '0 : i_data[i];
	end
	
end



endmodule : relu