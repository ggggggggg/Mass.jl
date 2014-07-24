using Mass, Mass.H5Helper, Base.Test

h = h5open("test.h5","w")
g = g_require(h, "g1")


@test a_read(g,"attribute",0) == 0
a_require(g,"attribute", 55)
@test a_read(g, "attribute", 0)==55
@test a_read(g, "attribute") == 55
@test a_require(g, "attribute", 55) == 55
@test_throws ErrorException a_require(g, "attribute", 4) == 55
a_update(g, "attribute", "string")
@test a_read(g, "attribute") == "string"
a_update(g, "new_attribute", 77)
@test a_read(g, "new_attribute")==77
for j = 1:100
a_update(g, "new_attribute", j)
@test a_read(g, "new_attribute",0)==j
end	
function f()
	j = a_read(g,"new_attribute",0)
	a_update(g, "new_attribute", j+1)
	@test j+1 == a_read(g,"new_attribute",0)
end
for j=1:100 f() end






d_update(g, "dataset", [1:10])
d_extend(g, "dataset", [11:20], 11:20)
@test g["dataset"][:] == [1:20]
d_extend(g, "dataset", [21:30])
@test g["dataset"][:] == [1:30]
d_update(g,"dataset",[1:10])
@test g["dataset"][:] == [1:10]