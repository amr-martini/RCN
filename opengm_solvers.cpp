#include <map>
#include <iostream>
#include <string>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/sparsemarray.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include <opengm/inference/visitors/visitors.hxx>

#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>
#include <opengm/inference/dualdecomposition/dddualvariableblock.hxx>
#include <opengm/inference/trws/trws_adsal.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/mqpbo.hxx>

#include <opengm/inference/external/trws.hxx>
#include <opengm/inference/external/ad3.hxx>
#include <opengm/inference/external/mplp.hxx>
#include <opengm/inference/external/srmp.hxx>
#include <srmp/SRMP.h>

// CANONICAL DEFS
typedef opengm::meta::TypeListGenerator<
    opengm::ExplicitFunction<double>,
    opengm::SparseMarray<double>
>::type FunctionTypeList; 
typedef opengm::DiscreteSpace<> Space;
typedef opengm::GraphicalModel<double, opengm::Multiplier, FunctionTypeList, Space> Model;
typedef opengm::Maximizer AccType;

// DD DEFS
typedef opengm::DDDualVariableBlock<marray::Marray<double> > DualBlockType;
typedef opengm::DDDualVariableBlock2<marray::Marray<double> > DualBlockType2;
typedef opengm::DualDecompositionBase<Model,DualBlockType>::SubGmType SubGmType;
typedef opengm::BeliefPropagationUpdateRules<SubGmType, AccType> UpdateRuleType;
typedef opengm::MessagePassing<SubGmType, AccType, UpdateRuleType, opengm::MaxDistance> InfType;

// INFERENCE ENGINE DEFS
typedef opengm::DualDecompositionSubGradient<Model,InfType,DualBlockType2> DualDecompositionSubGradient;
typedef opengm::DualDecompositionBundle<Model,InfType,DualBlockType2> DualDecompositionBundle_;
typedef opengm::ADSal<Model, AccType> ADSal;
typedef opengm::TRWSi<Model, AccType> TRWSi;
typedef opengm::MQPBO<Model, AccType> MQPBO_;

// EXTERNAL INFERENCE ENGINE DEFS
typedef opengm::external::TRWS<Model> TRWS;
typedef opengm::external::AD3Inf<Model, AccType> AD3_;
typedef opengm::external::MPLP<Model> MPLP_;
typedef opengm::external::SRMP<Model> SRMP_;

// PARAMETER DEFS
typedef DualDecompositionSubGradient::Parameter DualDecompositionSubGradient_Param;
typedef DualDecompositionBundle_::Parameter DualDecompositionBundle_Param;
typedef ADSal::Parameter ADSal_Param;
typedef TRWSi::Parameter TRWSi_Param;
typedef MQPBO_::Parameter MQPBO_Param;

// EXTERNAL PARAMETER DEFS
typedef TRWS::Parameter TRWS_Param;
typedef AD3_::Parameter AD3_Param;
typedef MPLP_::Parameter MPLP_Param;
typedef SRMP_::Parameter SRMP_Param;

using namespace std;

int main(int argc, char *argv[]) {

    if(argc != 4){
        cout << "Did not pass correct number of args. Skipping..." << endl;
        return 0;
    }

    const string model_pathfilename(argv[1]);
    const string alg_name(argv[2]);
    const int max_iterations = atoi(argv[3]);
    
    Model gm;
    opengm::hdf5::load(gm, model_pathfilename, "gm"); 
    
    if(alg_name.compare("AD3") == 0){
        AD3_Param param;
        param.steps_ = max_iterations;
        param.solverType_ = AD3_::AD3_LP;
        param.verbosity_ = 2;
        opengm::visitors::TimingVisitor<AD3_> visitor;
        AD3_ inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    else if(alg_name.compare("DDSG") == 0){
        DualDecompositionSubGradient_Param param;
        param.subPara_.maximumNumberOfSteps_ = max_iterations;
        param.useAdaptiveStepsize_ = true;
        param.useProjectedAdaptiveStepsize_ = true;
        opengm::visitors::TimingVisitor<DualDecompositionSubGradient> visitor;
        DualDecompositionSubGradient inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    // Compiling with DualDecompositionBlock throws compilation errors
    /*else if(alg_name.compare("DDB") == 0){
        DualDecompositionBundle_Param param;
        param.subPara_.maximumNumberOfSteps_ = max_iterations;
        opengm::visitors::TimingVisitor<DualDecompositionBundle_> visitor;
        DualDecompositionBundle_ inf_engine(gm, param);
        inf_engine.infer(visitor);
    }*/
    else if(alg_name.compare("ADSal") == 0) {
        const ADSal_Param param(max_iterations);
        opengm::visitors::TimingVisitor<ADSal> visitor;
        ADSal inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    else if(alg_name.compare("CMP") == 0) {
        SRMP_Param param;
        param.FullDualRelaxation_ = true;
        param.method = srmpLib::Energy::Options::CMP;
        param.iter_max = max_iterations;
        param.time_max = 1000000;
        param.verbose = true;
        opengm::visitors::TimingVisitor<SRMP_> visitor;
        SRMP_ inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    else if(alg_name.compare("SRMP") == 0){
        SRMP_Param param;
        param.FullDualRelaxation_ = true;
        param.method = srmpLib::Energy::Options::SRMP;
        param.iter_max = max_iterations;
        param.time_max = 1000000;
        param.verbose = true;
        opengm::visitors::TimingVisitor<SRMP_> visitor;
        SRMP_ inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    else if(alg_name.compare("MPLP") == 0){
        SRMP_Param param;
        param.FullDualRelaxation_ = true;
        param.method = srmpLib::Energy::Options::MPLP;
        param.iter_max = max_iterations;
        param.time_max = 1000000;
        param.verbose = true;
        opengm::visitors::TimingVisitor<SRMP_> visitor;
        SRMP_ inf_engine(gm, param);
        inf_engine.infer(visitor);
    }
    else {
        cout << "Passed alg name that wasn't understood. Skipping..." << endl;
        return 0;
    }
    
    // UNUSED
    //const DualDecompositionSubGradient_Param param;
    //const DualDecompositionBundle_Param param;
    //const TRWSi_Param param(20);
    //TRWS_Param param;
    //param.numberOfIterations_ = 20;
    //const MQPBO_Param param;
    //const MPLP_Param param;
    //opengm::hdf5::save(gm, "inpainting-n4.h5", "gm");

    return 0; 
} 
