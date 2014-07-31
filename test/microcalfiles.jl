module MicrocalFilesTest
using Mass, Base.Test
fname1 = "/a/b/c/a_chan1.ljh"
fname2 = "/a/b/c/b_chan3.ljh"
fname3 = "/a/b/otherdir/c_chan3.ljh"

@test hdf5_name_from_ljh(fname1) == "/a/b/c/a_mass.hdf5"
@test hdf5_name_from_ljh(fname1, fname2) == "/a/b/c/ab_mass.hdf5"
@test hdf5_name_from_ljh(fname1, fname2, fname3) == "/a/b/c/abc_mass.hdf5"

end