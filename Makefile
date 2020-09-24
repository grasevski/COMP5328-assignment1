zehu4485_ngra5777.zip: algorithm.py report/report.pdf
	[ -d data ] || (unzip resources/data.zip && rm -r __MACOSX)
	./$<
	mkdir -p zehu4485_ngra5777/code/algorithm zehu4485_ngra5777/code/data
	cp $< zehu4485_ngra5777/code/algorithm
	cp $(word 2,$^) zehu4485_ngra5777
	cd zehu4485_ngra5777 && zip -r ../$@ *
	rm -r zehu4485_ngra5777

report/report.pdf: report/report.tex report/report.sty
	cd report && pdflatex report.tex

clean:
	rm -rf zehu4485_ngra5777.zip report/report.aux report/report.log report/report.out report/report.pdf data
