// Local, bounded GPU trajectory recorder. No HTTP, no anatomy/1-D profile code.
// Input/output arrays are explicitly mapped to the source's XYZ C-order cells.
#include "lbm.hpp"
#include "info.hpp"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

template<class T> std::vector<T> read_array(const std::string& path, size_t n) {
    std::ifstream file(path,std::ios::binary); std::vector<T> data(n);
    if(!file.read(reinterpret_cast<char*>(data.data()),n*sizeof(T))) throw std::runtime_error("Missing or short input: "+path);
    if(file.peek()!=EOF) throw std::runtime_error("Oversized input: "+path);
    return data;
}
void write_array(const std::string& path,const std::vector<float>& data) {
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float))) throw std::runtime_error("Frame write failed");
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Usage: airway-gpu CASE_DIRECTORY");
        const std::string root=argv[1];
        uint nx,ny,nz,nfluid,intervals,stepsPerFrame,nfaces; float h,dt,nu,period,peakLattice,filter;
        std::ifstream config(root+"/input.txt");
        if(!(config>>nx>>ny>>nz>>nfluid>>intervals>>stepsPerFrame>>h>>dt>>nu>>period>>peakLattice>>nfaces>>filter)) throw std::runtime_error("Invalid configuration");
        const ulong total=ulong(nx)*ny*nz;
        if(!total||total>40000000||!nfluid||nfluid>total||intervals>512||!stepsPerFrame||!(h>0&&dt>0&&nu>0)||!(filter>=0&&filter<.5f)) throw std::runtime_error("Invalid limits");
        auto flags=read_array<uchar>(root+"/flags.u8",total);
        auto ids=read_array<uint>(root+"/cells.u32",nfluid);
        auto faces=read_array<int>(root+"/faces.i32",size_t(nfaces)*5);
        for(uint i:ids) if(i>=total||flags[i]!=0) throw std::runtime_error("Invalid fluid cell map");
        LBM lbm(nx,ny,nz,nu,period>0?2.0f*pif*dt/period:0.0f,peakLattice,filter); // fx: waveform angular frequency, fy: lattice pressure amplitude, fz: temporal momentum filter
        if(lbm.get_Nx()!=nx||lbm.get_Ny()!=ny||lbm.get_Nz()!=nz) throw std::runtime_error("Backend changed grid dimensions");
        for(ulong i=0;i<total;i++) {
            lbm.flags[i]=flags[i];
        }
        lbm.run(0u);
        auto start=std::chrono::steady_clock::now();
        std::filesystem::create_directories(root+"/frames");
        std::ofstream history(root+"/history.jsonl");
        std::vector<float> v(nfluid*3),p(nfluid);
        for(uint frame=0;frame<=intervals;frame++) {
            if(frame) lbm.run(stepsPerFrame);
            lbm.update_fields();lbm.u.read_from_device();lbm.rho.read_from_device();
            double maxSpeed=0,maxDensity=0;bool finite=true;
            for(uint j=0;j<nfluid;j++) {
                const uint i=ids[j];const float r=lbm.rho[i];
                float ux=lbm.u.x[i],uy=lbm.u.y[i],uz=lbm.u.z[i];
                finite=finite&&std::isfinite(r)&&std::isfinite(ux)&&std::isfinite(uy)&&std::isfinite(uz);
                maxSpeed=fmax(maxSpeed,sqrt(ux*ux+uy*uy+uz*uz));maxDensity=fmax(maxDensity,fabs(r-1));
                v[j*3]=ux*h/dt;v[j*3+1]=uy*h/dt;v[j*3+2]=uz*h/dt;
                p[j]=(r-1.f)/3.f*1.2f*h*h/(dt*dt);
            }
            if(!finite||maxSpeed>.17||maxDensity>.05) throw std::runtime_error("GPU trajectory failed finite/Mach/density checks at frame "+std::to_string(frame));
            write_array(root+"/frames/"+std::to_string(frame)+".velocity.f32",v);
            write_array(root+"/frames/"+std::to_string(frame)+".pressure.f32",p);
            std::vector<float> flux(nfaces);
            for(uint j=0;j<nfaces;j++) {
                const uint a=faces[j*5],b=faces[j*5+1],axis=faces[j*5+2];
                if(a>=total||b>=total||axis>2) throw std::runtime_error("Invalid boundary face map");
                // The reservoir velocity is reconstructed from the adjacent
                // fluid for pressure openings; velocity openings are imposed.
                const auto component=[&](uint i){return axis==0?lbm.u.x[i]:axis==1?lbm.u.y[i]:lbm.u.z[i];};
                flux[j]=component(a)*faces[j*5+3]*h*h*h/dt*1.e6f;
            }
            write_array(root+"/frames/"+std::to_string(frame)+".flux.f32",flux);
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            history<<std::setprecision(12)<<"{\"frame\":"<<frame<<",\"timeS\":"<<lbm.get_t()*double(dt)<<",\"computeSeconds\":"<<elapsed<<",\"maxMach\":"<<maxSpeed*sqrt(3.)<<",\"maxDensityDeviation\":"<<maxDensity<<"}"<<std::endl;
            std::cout<<"AIRWAY_FRAME "<<frame<<" "<<lbm.get_t()*double(dt)<<" "<<elapsed<<std::endl;
        }
        return 0;
    }catch(const std::exception& e) {std::cerr<<"AIRWAY_ERROR "<<e.what()<<std::endl;return 1;}
}
