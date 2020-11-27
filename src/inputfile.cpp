#include "inputfile.h"
#include "logger.h"
#include <scn/scn.h>
#include <fstream>
#include "util.h"

namespace tonto::io {

GaussianInputFile::GaussianInputFile(const std::string &filename)
{
    std::ifstream file(filename);
    parse(file);
}

GaussianInputFile::GaussianInputFile(std::istream &stream)
{
    parse(stream);
}

void GaussianInputFile::parse(std::istream &stream)
{
    using tonto::util::trim;
    std::string line;
    while(std::getline(stream, line)) {
        trim(line);
        if(line[0] == '%') parse_link0(line);
        else if(line[0] == '#') {
            parse_command_line(line);
            tonto::log::info("Found command line, breaking");
            break;
        }
    }
}

void GaussianInputFile::parse_link0(const std::string &line)
{
    using tonto::util::trim_copy;
    auto eq = line.find('=');
    std::string cmd = line.substr(1, eq - 1);
    std::string arg = line.substr(eq + 1);
    tonto::log::info("Found link0 command {} = {}", cmd, arg);
    link0_commands.push_back({std::move(cmd), std::move(arg)});
}

void GaussianInputFile::parse_command_line(const std::string &line)
{


}

}
