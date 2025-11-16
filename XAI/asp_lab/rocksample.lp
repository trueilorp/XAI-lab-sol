#include <incmode>.


#program base.

rock(0..4).
cell((0..10,0..10)).

good(0).
good(1).

loc(0, (2,5)).
loc(1, (5,2)).
loc(2, (8,6)).
loc(3, (1,9)).

at((0,0), 1). %location (0,0) at time 1


#program step(t).
%TO BE COMPLETED (action preconditions as aggregates; effects and constraints...)
%possible actions are east/north/south/west(t) + sample(R, t)




#program check(t).
%:- query(t), t<2. %debug purpose
:- query(t), not sampled(R,t), rock(R), good(R). %all good rocks must be sampled



#show east/1.
#show south/1.
#show north/1.
#show west/1.
#show sample/2.
%#show at/2.
