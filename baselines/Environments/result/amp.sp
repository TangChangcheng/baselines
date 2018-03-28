** Generated for: hspiceD
** Generated on: Dec 20 19:11:01 2017
** Design library name: bandgap_1
** Design cell name: amp
** Design view name: schematic
.GLOBAL vdd!

*.inc '../param/inparameter'
*.inc '../param/outparameter'

*.DC v1 700e-3 800e-3 20e-3

.TEMP 25.0
.OPTION
+ optlist=3
+ nomod=1
+ brief=1
+ INGOLD=0
+ runlvl=6
+ measout=1
+ measdgt=10
+ notop=1
+ nxx = 1
+ fast = 1
+ lis_new = 1
+ measfile = 1

.inc '/home/tangcc/_march/Documents/Project/library/hspice.mdl'
** Library name: bandgap_1
** Cell name: amp_1_1
** View name: schematic

m3 net19 vb0 vdd! vdd! pch_25 l=ll w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m5 net18 vb0 vdd! vdd! pch_25 l=ll w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m0 net16 vip net3 net4 pch_25 l=1e-6 w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m2 net3 vb0 vdd! net4 pch_25 l=ll w='wp1*1' m=2 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m4 vout2 vb3 net19 vdd! pch_25 l=ll w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m6 vout1 vb3 net18 vdd! pch_25 l=ll w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m1 net15 vip net3 net4 pch_25 l=ll w='wp1*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wp1' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wp1' pd='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(310e-9+1*wp1)' ps='(1-int(500e-3)*2)*(460e-9+2*wp1)+(2-int(1.0)*2)*(610e-9+3*wp1)' nrd='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' nrs='(1-int(500e-3)*2)*(155e-9/wp1)+(2-int(1.0)*2)*(155e-9/wp1)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m7 vout2 vb2 net16 0 nch_25 l=ll w='wn*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wn' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wn' pd='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(310e-9+1*wn)' ps='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(610e-9+3*wn)' nrd='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' nrs='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m9 net15 vb1 0 0 nch_25 l=ll w='wn*1' m=2 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wn' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wn' pd='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(310e-9+1*wn)' ps='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(610e-9+3*wn)' nrd='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' nrs='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m8 vout1 vb2 net15 0 nch_25 l=ll w='wn*1' m=1 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wn' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wn' pd='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(310e-9+1*wn)' ps='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(610e-9+3*wn)' nrd='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' nrs='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0
m10 net16 vb1 0 0 nch_25 l=ll w='wn*1' m=2 nf=1 sd=310e-9 ad='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*155e-9)*wn' as='((1-int(500e-3)*2)*230e-9+(2-int(1.0)*2)*305e-9)*wn' pd='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(310e-9+1*wn)' ps='(1-int(500e-3)*2)*(460e-9+2*wn)+(2-int(1.0)*2)*(610e-9+3*wn)' nrd='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' nrs='(1-int(500e-3)*2)*(155e-9/wn)+(2-int(1.0)*2)*(155e-9/wn)' sa=230e-9 sb=230e-9 sca=0 scb=0 scc=0

*.model m10 OPT METHOD=BISECTION ITROPT=20 dynacc=1 relout=1e20
v10 vb3 0 DC=vb3
v9 vb2 0 DC=vb2
v8 vb1 0 DC=vb1
v7 vb0 0 DC=vb0
v1 vip 0 DC=vin ac=1
v0 vdd! 0 DC=1.8


* Simulation
.inc "/home/tangcc/_march/Documents/Project/baselines/baselines/Environments/param/outparameter"
.inc "/home/tangcc/_march/Documents/Project/baselines/baselines/Environments/param/inparameter"
*.sweep vb1 0.1v 0.0005v 1.8v
.END
