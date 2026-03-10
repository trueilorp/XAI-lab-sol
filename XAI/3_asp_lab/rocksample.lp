#include <incmode>.


#program base.

rock(0..4).
cell((0..10,0..10)). %our grid

good(0). %rock 0 is good
good(1). %rock 1 is good

loc(0, (2,5)). %location of the rocks
loc(1, (5,2)).
loc(2, (8,6)).
loc(3, (1,9)).

at((0,0), 1). %location (0,0) at time 1


#program step(t).
%TO BE COMPLETED (action preconditions as aggregates; effects and constraints...)
%possible actions are east/north/south/west(t) + sample(R, t)
%preconditions on movements not needed, already done in the grounding step cy clingo
#program step(t).

% exactly one action
1 { north(t); south(t); east(t); west(t); sample(R,t) : rock(R) } 1.

% movement with boundary checks
at((X+1,Y),t+1) :- east(t), at((X,Y),t), cell((X+1,Y)).
at((X-1,Y),t+1) :- west(t), at((X,Y),t), cell((X-1,Y)).
at((X,Y+1),t+1) :- north(t), at((X,Y),t), cell((X,Y+1)).
at((X,Y-1),t+1) :- south(t), at((X,Y),t), cell((X,Y-1)).

% inertia - no action
at(P,t+1) :- at(P,t), not north(t), not south(t), not east(t), not west(t).

% exactly one position - agent can be in only one position
:- not 1 { at(P,t) : cell(P) } 1.

% sampling precondition
:- sample(R,t), at(P,t), loc(R,P2), P != P2.

% sampling effects + inertia
sampled(R,t+1) :- sample(R,t).
sampled(R,t+1) :- sampled(R,t).

% avoid re-sampling
:- sample(R,t), sampled(R,t).

% avoid oscillations, so I I was at south, do not come back at south
:- north(t), south(t-1).
:- south(t), north(t-1).
:- east(t), west(t-1).
:- west(t), east(t-1).


#program check(t).
%:- query(t), t<2. %debug purpose, stop if you have the time istance >=2

% Reject any model where a good rock is not sampled.
:- query(t), not sampled(R,t), rock(R), good(R). % all good rocks must be sampled, the check is a a constrain, check if some good rock are not sampled


%show command let you decide what to output with clingo, just print the actions
#show east/1.
#show south/1.
#show north/1.
#show west/1.
#show sample/2.
#show at/2.