

function peakassign(locations_arb, locations_true)
# locations_arb is an array of possible peak locations in arbitrary units, assumed to have an unknown, 
# nonlinear (but monotonic and smooth) relationship to locations_true
# locations_arb may contain more peaks than locations_true, including some spurious peaks
# the goal is to find the best possible set of assignments for entries in locations_arb to 
# correspond to locations_true by examining all possible assignments and calcualting some figure of merit
@assert(issorted(locations_arb))
@assert(issorted(locations_true))
@assert(length(locations_arb)>=length(locations_true))
combos = combinations(locations_arb, length(locations_true))
length(combos)==1 && return first(combos)
@assert(length(locations_true)>=3)

best_combo = first(combos)
best_fom = realmax(Float64)
for c in combos
	fom = peakassign_fig_of_merit(c, locations_true)
	if fom < best_fom
		best_fom = fom
		best_combo = c
	end
end
best_combo, best_fom
end

function peakassign_fig_of_merit_old(x, y)
	# average fractional error trying to predict the y value of the middle of 3 points using the slope
	# from the outer two points and the x value of the middle point
	err = 0.0
	for i = 1:length(y)-2
		yip1_prediction = y[i]+(y[i+2]-y[i])*(x[i+1]-x[i])/(x[i+2]-x[i])
		err += abs(1-y[i+1]/yip1_prediction)
	end
	err /= length(y-2)
end 

function peakassign_fig_of_merit(x, y)
	std(log(y)./log(x))
end


trues =[1000 3000 4000 6000]'
arbs = trues.^1.1
arbs2 = sort(vec([arbs, arbs+500]))

c, fom = peakassign(arbs2, trues)