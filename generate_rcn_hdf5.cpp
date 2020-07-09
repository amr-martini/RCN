#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

typedef opengm::meta::TypeListGenerator<
    opengm::ExplicitFunction<double>
>::type FunctionTypeList; 

typedef opengm::DiscreteSpace<> Space;
typedef opengm::Maximizer AccType;
typedef opengm::Multiplier DecompType;

typedef opengm::GraphicalModel<
    double, 
    DecompType,
    FunctionTypeList, 
    Space
> RCN;

// 0 means OUT, so must use type1
// 1 means IN, so must use type2
unsigned char edge_patches[32][3][3] = {
    
    // R0
    {{0,0,1},
     {0,0,1},
     {0,0,1}},

    // R0
    {{1,1,0},
     {1,1,0},
     {1,1,0}},

    // R1
    {{1,0,0},
     {1,0,0},
     {1,0,0}},

    // R1
    {{0,1,1},
     {0,1,1},
     {0,1,1}},

    // R2
    {{1,1,1},
     {0,0,0},
     {0,0,0}},

    // R2
    {{0,0,0},
     {1,1,1},
     {1,1,1}},

    // R3
    {{1,1,1},
     {1,1,1},
     {0,0,0}},

    // R3
    {{0,0,0},
     {0,0,0},
     {1,1,1}},

    // R4
    {{1,1,0},
     {1,1,1},
     {1,1,1}},

    // R4    
    {{0,0,1},
     {0,0,0},
     {0,0,0}},

    // R5
    {{1,0,0},
     {1,1,0},
     {1,1,1}},

    // R5
    {{0,1,1},
     {0,0,1},
     {0,0,0}},

    // R6
    {{0,0,0},
     {1,0,0},
     {1,1,0}},

    // R6
    {{1,1,1},
     {0,1,1},
     {0,0,1}},

    // R7
    {{0,0,0},
     {0,0,0},
     {1,0,0}},

    // R7
    {{1,1,1},
     {1,1,1},
     {0,1,1}},

    // R8
    {{1,0,0},
     {1,0,0},
     {1,1,0}},

    // R8
    {{0,1,1},
     {0,1,1},
     {0,0,1}},

    // R9
    {{0,1,1},
     {0,0,1},
     {0,0,1}},

    // R9
    {{1,0,0},
     {1,1,0},
     {1,1,0}},

    // R10
    {{1,1,1},
     {0,0,1},
     {0,0,0}},

    // R10
    {{0,0,0},
     {1,1,0},
     {1,1,1}},

}


/*
// if feature is inner-inner (of different inner types), check this array for type1 usage
// if feature is not inner-inner (of different inner types), then always use type2 edge
unsigned char horizontal_canvas_edges[10][3][2] = {

    // R0
    {{2,1},
     {2,1},
     {2,1}},

    // R1
    {{1,2},
     {1,2},
     {1,2}},

    // R2
    {{2,2},
     {2,2},
     {2,2}},

    // R3
    {{2,2},
     {2,2},
     {2,2}},

    // R4
    {{2,1},
     {2,2},
     {2,2}},

    // R5
    {{1,2},
     {2,1},
     {2,2}},

    // R6
    {{2,2},
     {1,2},
     {2,1}},

    // R7
    {{2,2},
     {2,2},
     {1,2}},

    // R8
    {{1,2},
     {1,2},
     {2,1}},

    // R9
    {{1,2},
     {2,1},
     {2,1}},
}


// if feature is inner-inner (of different inner types), check this array for type1 usage
// if feature is not inner-inner (of different inner types), then always use type2 edge
unsigned char vertical_edges[10][2][3] = {
    
    // R0
    {{2,2,2},
     {2,2,2}},

    // R1
    {{2,2,2},
     {2,2,2}},

    // R2
    {{1,1,1},
     {2,2,2}},

    // R3
    {{2,2,2},
     {1,1,1}},

    // R4
    {{2,2,1},
     {2,2,2}},

    // R5
    {{2,1,2},
     {2,2,1}},

    // R6
    {{1,2,2},
     {2,1,2}},

    // R7
    {{2,2,2},
     {1,2,2}},

    // R8
    {{2,2,2},
     {2,1,2}},

    // R9
    {{2,1,2},
     {2,2,2}},
}*/

int main(int argc, char *argv[]){
    
    
    
    // ------ IMAGE LOAD ------
    
    if(argc < 2){
        cout << "Did not pass correct number of args. Skipping..." << endl;
        return 0;
    }
        
    Mat image;
    image = imread(argv[1], 0);   // Read the file

    if(!image.data )   // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    
    CV_Assert(image.depth() == CV_8U);

    cout << argv[1] << ": " << image.rows << " | " << image.cols << endl;
    
    int canvas_dim = image.rows < image.cols ? image.rows : image.cols;
    canvas_dim = (canvas_dim/3);
    
    int power_of_2 = 1;
    while(power_of_2*2 < canvas_dim)
        power_of_2 *= 2;
    
    const int canvas_rows = power_of_2*3;
    cout << "Shortening to <" << canvas_rows << " | " << canvas_rows << endl;
    
    

    
    // ----- HYPERPARAMETERS ------

    int canvas_num_labels = 2*32;
    int downscale_factor = 256/(canvas_num_labels/2); // don't change
    int max_feature_depth = 32;
    double mixture_parameter = 0.1;
    double radius_parameter = 3;
    double canvas_edgemap_disagreement_penalty = 0.1;
    double background_mixture_value = (1/((double)(canvas_num_labels/2))); // don't change
    int alpha; // note that here it is an int, and here it is half the value as defined in the paper
    double beta;
    double gamma;

    
    
    // ----- CONSTRUCT LABEL SPACE + MODEL ------
    
    int current_row_dim = canvas_rows/3;
    int current_col_dim = canvas_rows/3;
    int current_feature_dim = max_feature_depth;
    
    if(current_col_dim < current_feature_dim)
    {
        cout << "canvas dim is smaller than feature dim" << endl;
        return 0;
    }
        
    Space space;
    
    // canvas variables
    for(int i = 0; i < canvas_rows; i++){
        for(int j = 0; j < canvas_rows; j++){
            space.addVariable(canvas_num_labels);
        }
    }
    
    const int edgemap_vars_offset = canvas_rows*canvas_rows;
    
    // edge map
    for(int f = 0; f < current_feature_dim; f++){
        for(int i = 0; i < current_row_dim; i++){
            for(int j = 0; j < current_col_dim; j++){
                space.addVariable(2);
            }
        }
    }
    
    current_row_dim /= 2;
    const int edgemap_channel_multiplier = current_row_dim*current_col_dim;
    const int hierarchy_vars_offset = edgemap_vars_offset + current_feature_dim*current_row_dim*current_col_dim;
    
    // alternating pool - feature layers
    int do_pool_layer = 1;
    while(current_row_dim > 1){
        if(do_pool_layer & 1){
            // add pool layer

            for(int f = 0; f < current_feature_dim; f++){
                for(int i = 0; i < current_row_dim; i++){
                    for(int j = 0; j < current_col_dim; j++){
                        space.addVariable(3);
                    }
                }
            }
            
            current_col_dim /= 2;
            current_feature_dim /= current_feature_dim > 1 ? 2 : 1;
        }
        else{
            // add feature layer

            for(int f = 0; f < current_feature_dim; f++){
                for(int i = 0; i < current_row_dim; i++){
                    for(int j = 0; j < current_col_dim; j++){
                        space.addVariable(2);
                    }
                }
            }
            
            current_row_dim /= 2;
        }

        do_pool_layer += 1;
    }
    
    // add pass through feature layer
    space.addVariable(2);
    space.addVariable(2);
    
    // add top level pool variable
    space.addVariable(2);
    
    RCN gm(space);
    
    
    // ----- FUNCTIONS ------
  
    
    // ---- functions for canvas-edgemap edges ----
    
    // first half of canvas states mean IN; second half of states mean OUT
    // type1 is used to force a canvas variable to be OUT 
    // type2 is used to force a canvas variable to be IN
    size_t f_canvas_edgemap_shape[] = {canvas_num_labels, 2};
    
    opengm::ExplicitFunction<double> f_canvas_edgemap_type1(f_canvas_edgemap_shape, f_canvas_edgemap_shape+2, 1);
    for(int i = 0; i < canvas_num_labels/2; i++){
        f_canvas_edgemap_type1(i,1) = canvas_edgemap_disagreement_penalty;
    }
    RCN::FunctionIdentifier fid_canvas_edgemap_type1 = gm.addFunction(f_canvas_edgemap_type1);

    opengm::ExplicitFunction<double> f_canvas_edgemap_type2(f_canvas_edgemap_shape, f_canvas_edgemap_shape+2, 1);
    for(int i = canvas_num_labels/2; i < canvas_num_labels; i++){
        f_canvas_edgemap_type2(i,1) = canvas_edgemap_disagreement_penalty;
    }
    RCN::FunctionIdentifier fid_canvas_edgemap_type2 = gm.addFunction(f_canvas_edgemap_type2);

    
    // ---- functions for canvas-image edges ----
    
    size_t f_canvas_image_shape[] = {canvas_num_labels};
    vector<RCN::FunctionIdentifier> fids_canvas_image;
    // iterate over pixels, save functions ids to vector
    for(int i = 0; i < canvas_rows; ++i){
        for (int j = 0; j < canvas_rows; ++j){
            unsigned char grayscale_value = image.at(i,j);
            grayscale_value /= downscale_factor;
            opengm::ExplicitFunction<double> f_canvas_image(f_canvas_image_shape, f_canvas_image_shape+1, background_mixture_value);
            
            int lower_bound = max(grayscale_value-radius_parameter, 0);
            int upper_bound = min(grayscale_value+radius_parameter, canvas_num_labels/2 - 1);
            double range = upper_bound - lower_bound;
            for(int k = lower_bound; k <= upper_bound; k++){
                f_canvas_image(k) = mixture_parameter*background_mixture_value + (1-mixture_parameter)*(1/range);
                f_canvas_image(k+canvas_num_labels/2) = mixture_parameter*background_mixture_value + (1-mixture_parameter)*(1/range);
            }
            
            fids_canvas_image.push_back(gm.addFunction(f_canvas_image));
        }
    }
    

    // ---- functions for canvas-canvas edges ----
    
    // type1 for edges across different inner regions
    // type2 for edges across inner-outer regions, inner-inner regions of same type, and outer-outer regions
    size_t f_canvas_canvas_shape[] = {canvas_num_labels, canvas_num_labels};
    
    /*
    opengm::ExplicitFunction<double> f_canvas_canvas_type1(f_canvas_canvas_shape, f_canvas_canvas_shape+2, 1);
    for(int i = 0; i < canvas_num_labels/2; i++){
        for(int j = i-alpha; j <= i+alpha; j++){
            if(j < 0 || j >= canvas_num_labels/2)
                continue;
            f_canvas_canvas_type1(i,j) = beta;
        }
    }
    RCN::FunctionIdentifier fid_canvas_canvas_type1 = gm.addFunction(f_canvas_canvas_type1); */
    
    opengm::ExplicitFunction<double> f_canvas_canvas_type2(f_canvas_canvas_shape, f_canvas_canvas_shape+2, 1);
    for(int i = 0; i < canvas_num_labels/2; i++){
        for(int j = canvas_num_labels/2; j < canvas_num_labels; j++){
            f_canvas_canvas_type2(i,j) = gamma;
            if(abs(i+canvas_num_labels/2-j) > 2*alpha){
                f_canvas_canvas_type2(i,j-canvas_num_labels/2) = beta;
            }
        }
    }
    for(int i = canvas_num_labels/2; i < canvas_num_labels; i++){
        for(int j = 0; j < canvas_num_labels/2; j++){
            f_canvas_canvas_type2(i,j) = gamma;
            if(abs(i-canvas_num_labels/2-j) > 2*alpha){
                f_canvas_canvas_type2(i-canvas_num_labels/2,j) = beta;
            }
        }
    }

    for(int i = 0; i < canvas_num_labels/2; i++){
        for(int j = i-alpha; j <= i+alpha; j++){
            if(j < 0 || j >= canvas_num_labels/2)
                continue;
            f_canvas_canvas_type2(i+canvas_num_labels/2,j) = beta*gamma;
            f_canvas_canvas_type2(i,j+canvas_num_labels/2) = beta*gamma;
        }
    }
    RCN::FunctionIdentifier fid_canvas_canvas_type2 = gm.addFunction(f_canvas_canvas_type2);
    

    // ---- functions for feature-pool edges (basically pass-through) ----
    
    // type1 means feature is activated if pool takes on state 1 (left-child)
    // type2 means feature is activated if pool takes on state 2 (right-child)
    size_t f_feature_pool_shape[] = {2,3};
    opengm::ExplicitFunction<double> f_feature_pool_type1(f_feature_pool_shape,f_feature_pool_shape+2,1);
    f_feature_pool_type1(0,1) = 0;
    f_feature_pool_type1(1,0) = 0;
    f_feature_pool_type1(1,2) = 0;
    RCN::FunctionIdentifier fid_feature_pool_type1 = gm.addFunction(f_feature_pool_type1);
    
    opengm::ExplicitFunction<double> f_feature_pool_type2(f_feature_pool_shape,f_feature_pool_shape+2,1);
    f_feature_pool_type2(0,2) = 0;
    f_feature_pool_type2(1,0) = 0;
    f_feature_pool_type2(1,1) = 0;
    RCN::FunctionIdentifier fid_feature_pool_type2 = gm.addFunction(f_feature_pool_type2);
    
    
    // ---- functions for pool-feature edges ----
    size_t f_pool_feature_shape[] = {3,2};
    opengm::ExplicitFunction<double> f_pool_feature(f_pool_feature_shape,f_pool_feature_shape+2,0);
    f_pool_feature(0,0) = 1;
    f_pool_feature(1,1) = 1;
    f_pool_feature(2,1) = 1;
    RCN::FunctionIdentifier fid_pool_feature = gm.addFunction(f_pool_feature);
    
    
    // ---- functions for top-level edges ----

    // type1 means feature is activated if pool takes on state 0 (left-child)
    // type2 means feature is activated if pool takes on state 1 (right-child)
    size_t f_top_level_shape[] = {2,2};
    opengm::ExplicitFunction<double> f_top_level_type1(f_top_level_shape,f_top_level_shape+2,1);
    opengm::ExplicitFunction<double> f_top_level_type2(f_top_level_shape,f_top_level_shape+2,1);
    f_top_level_type1(0,0) = 0;
    f_top_level_type1(1,1) = 0;
    f_top_level_type2(0,1) = 0;
    f_top_level_type2(1,0) = 0;
    RCN::FunctionIdentifier fid_top_level_type1 = gm.addFunction(f_top_level_type1);
    RCN::FunctionIdentifier fid_top_level_type2 = gm.addFunction(f_top_level_type2);
    
    
    
    
    // ----- ADD EDGES (IE FACTORS) ------
    
    int current_row_dim = canvas_rows/3;
    int current_col_dim = canvas_rows/3;
    int current_feature_dim = max_feature_depth;
    
    // canvas variables
    for(int i = 0; i < canvas_rows; i++){
        for(int j = 0; j < canvas_rows; j++){
            size_t vi = i*canvas_rows + j;
            size_t vj_right_neighbor = i*canvas_rows + j+1;
            size_t vk_down_neighbor = (i+1)*canvas_rows + j;
            
            
            // canvas-image unary
            gm.addFactor(fids_canvas_image[vi], &vi, &vi+1);
            
            
            // canvas-canvas factors
            if(j+1 < canvas_rows){
                size_t horizontal_edge[] = {vi, vj_right_neighbor};
                gm.addFactor(fid_canvas_canvas_type2, horizontal_edge, horizontal_edge+2);
            }
            if(i+1 < canvas_rows){
                size_t vertical_edge[] = {vi, vk_down_neighbor};
                gm.addFactor(fid_canvas_canvas_type2, vertical_edge, vertical_edge+2);
            }
            
            
            // canvas-edgemap factors
            for(int f = 0; f < current_feature_dim; f++){
                const int feature_i = i/3;
                const int feature_j = j/3;
                size_t feature_v = edgemap_vars_offset + \
                                   f*edgemap_channel_multiplier + \ 
                                   feature_i*edgemap_i_multiplier + \
                                   feature_j;
                size_t canvas_feature_edge[] = {vi, feature_v};
                
                unsigned char use_type2 = edge_patches[f][i % 3][j % 3];
                if(use_type2){
                    gm.addFactor(fid_canvas_edgemap_type2, canvas_feature_edge, canvas_feature_edge+2);
                }
                else {
                    gm.addFactor(fid_canvas_edgemap_type1, canvas_feature_edge, canvas_feature_edge+2);
                }
            }
        }
    }
    
    int current_layer_offset = edgemap_vars_offset;
    int current_parent_offset = hierarchy_vars_offset;

    // edge map
    for(int f = 0; f < current_feature_dim; f++){
        for(int i = 0; i < current_row_dim; i++){
            for(int j = 0; j < current_col_dim; j++){
                size_t current_channel_multiplier = current_row_dim*current_col_dim;
                size_t edgemap_v = current_layer_offset + \
                                   f*current_channel_multiplier + \ 
                                   i*current_col_dim + \
                                   j;
                size_t pool_i = i/2;
                size_t pool_j = j;
                size_t pool_f = f;
                size_t pool_v = current_parent_offset +\
                                pool_f*(current_channel_multiplier/2) +\
                                pool_i*current_col_dim +\
                                pool_j;
                size_t feature_pool_edge[] = {edgemap_v, pool_v};
                
                if(i & 1 == 0){
                    gm.addFactor(fid_feature_pool_type1, feature_pool_edge, feature_pool_edge+2);
                }
                else{
                    gm.addFactor(fid_feature_pool_type2, feature_pool_edge, feature_pool_edge+2);
                }
            }
        }
    }
    
    current_row_dim /= 2;
    current_layer_offset = current_parent_offset;
    current_parent_offset += current_feature_dim*current_row_dim*current_col_dim;
    
    // alternating pool - feature layers
    int do_pool_layer = 1;
    while(current_row_dim > 1){
        if(do_pool_layer & 1){
            // add pool layer
            size_t current_channel_multiplier = current_row_dim*current_col_dim;

            for(int f = 0; f < current_feature_dim; f++){
                for(int i = 0; i < current_row_dim; i++){
                    for(int j = 0; j < current_col_dim; j++){
                        size_t pool_v = current_layer_offset +\
                                        f*current_channel_multiplier +\
                                        i*current_col_dim +\
                                        j;
                        size_t feature_f = current_feature_dim > 1 ? f/2 : 1;
                        size_t feature_i = i;
                        size_t feature_j = j/2;
                        size_t feature_v = current_parent_offset +\
                                           feature_f*(current_channel_multiplier/2) +\
                                           feature_i*(current_col_dim/2) +\
                                           feature_j;
                        size_t pool_feature_edge[] = {pool_v, feature_v};
                        
                        gm.addFactor(fid_pool_feature, pool_feature_edge, pool_feature_edge+2);
                    }
                }
            }
            
            current_col_dim /= 2;
            current_feature_dim /= current_feature_dim > 1 ? 2 : 1;
            current_layer_offset = current_parent_offset;
            current_parent_offset += current_feature_dim*current_row_dim*current_col_dim;
        }
        else{
            // add feature layer

            for(int f = 0; f < current_feature_dim; f++){
                for(int i = 0; i < current_row_dim; i++){
                    for(int j = 0; j < current_col_dim; j++){
                        size_t current_channel_multiplier = current_row_dim*current_col_dim;
                        size_t feature_v = current_layer_offset + \
                                           f*current_channel_multiplier + \ 
                                           i*current_col_dim + \
                                           j;
                        size_t pool_i = i/2;
                        size_t pool_j = j;
                        size_t pool_f = f;
                        size_t pool_v = current_parent_offset +\
                                        pool_f*(current_channel_multiplier/2) +\
                                        pool_i*current_col_dim +\
                                        pool_j;
                        size_t feature_pool_edge[] = {feature_v, pool_v};

                        if(i & 1 == 0){
                            gm.addFactor(fid_feature_pool_type1, feature_pool_edge, feature_pool_edge+2);
                        }
                        else{
                            gm.addFactor(fid_feature_pool_type2, feature_pool_edge, feature_pool_edge+2);
                        }
                    }
                }
            }
            
            current_row_dim /= 2;
            current_layer_offset = current_parent_offset;
            current_parent_offset += current_feature_dim*current_row_dim*current_col_dim;
        }

        do_pool_layer += 1;
    }
    
    // add pass through feature layer
    current_parent_offset = current_layer_offset + 2;
    
    size_t pool_v = current_layer_offset;
    size_t feature_v = current_parent_offset;
    size_t pool_feature_edge[] = {pool_v, feature_v};

    gm.addFactor(fid_pool_feature, pool_feature_edge, pool_feature_edge+2);
    
    pool_v++;
    feature_v++;
    pool_feature_edge = {pool_v, feature_v};
    
    gm.addFactor(fid_pool_feature, pool_feature_edge, pool_feature_edge+2);
    
    // add top level pool variable connections
    pool_v += 3;
    feature_v--;
    size_t feature_pool_edge[] = {feature_v, pool_v};
    
    gm.addFactor(fid_feature_pool_type1, feature_pool_edge, feature_pool_edge+2);
    
    feature_v++;
    feature_pool_edge = {feature_v, pool_v};
    
    gm.addFactor(fid_feature_pool_type2, feature_pool_edge, feature_pool_edge+2);

    
    
    
    // ------- SAVE -------
    
    
    opengm::hdf5::save(gm, "rcn-gm.h5", "gm");
    
    return 0;
}
