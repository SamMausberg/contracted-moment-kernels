// SPDX-License-Identifier: Apache-2.0
#include "imported.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("Usage: cmk_imported_check fixture.txt");
 std::ifstream f(argv[1]);int rows=0,B=0;f>>rows>>B;
 if(!f||rows<1||rows>100000||B<1||B>100000)throw std::runtime_error("Invalid dimensions");
 int accepted=0,exact_accepted=0;
 for(int row=0;row<rows;++row){double l=0,u=0;int expected=0;f>>l>>u>>expected;
  std::vector<cmk::ImportedBox> boxes(B);
  for(auto&b:boxes)f>>b.zlo>>b.zhi>>b.mlo>>b.mhi>>b.centerlo>>b.centerhi;
  if(!f)throw std::runtime_error("Truncated/malformed fixture");
  bool pass=cmk::check_imported(boxes.data(),B,l,u);accepted+=pass;exact_accepted+=expected;
  if(pass&&!expected)throw std::runtime_error("False certificate vs rational residual oracle");
 }
 std::cout<<"{\"imported_checker_pass\":true,\"rows\":"<<rows
  <<",\"host_certified_rows\":"<<accepted<<",\"rational_certified_rows\":"<<exact_accepted<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
