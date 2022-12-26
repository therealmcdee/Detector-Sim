# Detector-Sim
Analyzing programs for Scintillator SiPM Detector
This repository includes .mac files for GEANT4 GEARS and some analysis pyROOT code used for determining the position of an incident beta particle on a plastic scintillator detector.

Files:

Sr_90_spec.py -- display Strontium-90 beta emission spectrum

write_*90_spec.py -- puts beta emission spectrum into macro file for GEANT GEARS ( * == Sr,Y)

*90.mac -- GEANT GEARS macro file loaded with emission spectrum of strontium/yttrium-90

run_pos_scan.mac & rm_test.mac -- macro files; electron strike point on scintillating target 

rm2_test.mac -- macro file loaded with Sr90 beta emission spectrum


