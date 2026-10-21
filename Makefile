PY      ?= python3
IVERILOG ?= iverilog
VVP     ?= vvp

RTL := rtl/pe.sv
TB  := tb/tb_pe.sv

.PHONY: test pytest sim roofline clean

test: pytest sim

pytest:
	$(PY) -m pytest -q

sim: sim/pe.vvp
	$(VVP) -n sim/pe.vvp

sim/pe.vvp: $(RTL) $(TB)
	@mkdir -p sim
	$(IVERILOG) -g2012 -Wall -o $@ $(RTL) $(TB)

roofline:
	$(PY) -m analysis.roofline --out docs/figures/roofline_t4.png

clean:
	rm -rf sim .pytest_cache **/__pycache__
