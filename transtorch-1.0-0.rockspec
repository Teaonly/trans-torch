package = "transTorch"
version = "1.0-0"

source = {
   url = "git://github.com/teaonly/transTorch",
   tag = "master"
}

description = {
   summary = "Parameter transformer between Torch and Caffe",
   detailed = [[
A simple solution model converter between Torch and Caffe.
We don't create network, just transfom weight between two already model. 
   ]],
   homepage = "https://github.com/teaonly/transTorch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
