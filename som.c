



#include "m_pd.h" //importa as funcões prontas do pd
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_blas_types.h"
#include "gsl/gsl_math.h"



#define MAXNEURON 512
#define MAXVECTOR 3
#define NET 2
#define MAXTOPO 8
// #define IN 40
#define MAXRAND 268435457
#define DENRAND 268435456

//define uma nova classe
static t_class *som_class;


//definição da estrutura do objeto - tudo que o objeto precisa (inlets, outlets, variaveis)

typedef struct _som {
    t_object  x_obj;
    t_float   x_neurons; //numero de neuronios
    t_float   x_vec; // dimensão dos vetores
    t_float   x_d; //variável que armazena distância euclidiana
    t_float   x_netsize[NET];


    gsl_matrix *m; // matriz com os pesos dos neurônios
    gsl_vector *data; //array para armazenar os dados de entrada
    gsl_vector *d;
    gsl_vector *dist; //array para armazenar as distâncias 
    gsl_vector *win; // distâncias ordenadas da maior para a menar
    gsl_vector *wv; //pesos do neurônio vencedor atualizado 
    gsl_vector *vizi; // distâncias dos neurônios vizinhos
    gsl_vector *index_vizi;
    gsl_vector *errodiff;
    gsl_vector *errosum;




//armazena os pesos x, y e z para saída no outlet 1
    t_atom     *x_peso_outx; 
    t_atom     *x_peso_outy;
    t_atom     *x_peso_outz;

    t_atom     *x_peso_win;
    t_atom     *x_index; // índice do neurônio vencedor

// armazena os valores dos parametros atualizados para saída no outle 2
    t_atom     *x_current_learn;
    t_atom     *x_current_iter;
    t_atom     *x_current_raiovizi;
    t_atom     *x_erro; //erro quadrático

    t_canvas *x_canvas;

    
//parametros internos da rede
    t_float   x_learn; //taxa de aprendizado
    t_float   x_input; // tamanho dos dados de entrada
    t_float   x_epoc; //numero de iterações da rede
    t_float   x_topo; //quantidade de neurônios vizinhos
    t_float   x_raiovizi; //raio de vizinhanção 
    t_float   x_count; //contador de dados que estão entrando na rede
    t_float   x_iter; //contador de iterações
    t_float   x_trainingmode;
    
    
//outlets do objeto
    t_outlet *out1; //pesos da rede
    t_outlet *out2; //peso e posição do neurônio vencedor
    t_outlet *out3; //iteração atual, learning atual, raio de vizinhança atual, erro quadrático
} t_som; 



//------- recebe lista com número de neurônios e tamanho do vetor de pesos, e aloca memórica necessária para a matriz e os vetores-------//
void som_neuron_size(t_som *x, t_symbol *s, int argc, t_atom *argv){ 

    int i;
    int j;

    
//verifica se a lista recebida tem tamnaho 2 e atribui os valores para cada variável
    if(argc == 2){
        for (i = 0; i < argc; i++){  
            x->x_netsize[i] = argv[i].a_w.w_float; 
        }
    x->x_neurons = x->x_netsize[0];

    if(x->x_netsize[1] <= MAXVECTOR){
        x->x_vec = x->x_netsize[1];
    }
    else{
        error("the maximum size of vectors has been exceeded");
    }

//desaloca memória
    gsl_matrix_free (x->m);
    gsl_vector_free (x->data);
    gsl_vector_free (x->d);
    gsl_vector_free (x->dist);
    gsl_vector_free (x->win);
    gsl_vector_free (x->wv);
    gsl_vector_free (x->errodiff);
    freebytes(x->x_peso_outx, x->x_neurons * sizeof(t_atom)); 
    freebytes(x->x_peso_outy, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_outz, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_win, x->x_vec * sizeof(t_atom));
    

//aloca memórica com os novos valores
    x->m = gsl_matrix_alloc (x->x_neurons, x->x_vec);
    x->data = gsl_vector_alloc (x->x_vec);
    x->d = gsl_vector_alloc (x->x_vec);
    x->dist = gsl_vector_alloc (x->x_neurons);
    x->win = gsl_vector_alloc (x->x_neurons);
    x->wv = gsl_vector_alloc (x->x_vec);
    x->errodiff = gsl_vector_alloc (x->x_neurons);
    x->x_peso_outx = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom)); 
    x->x_peso_outy = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    x->x_peso_outz = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    x->x_peso_win = (t_atom *)getbytes(x->x_vec * sizeof(t_atom));
    
    

}
    else{
        error("please, provide the number of neurons and vectors for the network");
    }

post("neurons: %0.1f, vectors: %0.1f", x->x_neurons, x->x_vec);

for (i = 0; i < x->x_neurons; i++){
        for (j = 0; j < x->x_vec; j++){
            gsl_matrix_set (x->m, i, j, (double)rand()/RAND_MAX);
            // post("neuron(%d,%d): %g\n", i, j, gsl_matrix_get (x->m, i, j));
        }

    } 

}

//---------------- recebe a taxa de aprendizado ------------------//
static void som_learn (t_som *x, t_floatarg le){ 
    
  if(le >= -1 && le <= 1){
    x->x_learn = le;
    }
}

//---------------- recebe tamanho do dado de entrada ------------------//
static void som_input (t_som *x, t_floatarg in){
    gsl_vector_free (x->errosum);

    if(in == floor(in) && in > 0){
        x->x_input = (int)in;
    }
    else{
        error("please, provide a interger value");
    }
    x->errosum = gsl_vector_alloc (x->x_input);

}

// ---------------- recebe quantidade de épocas ----------------------//
static void som_epoc (t_som *x, t_floatarg ep){

    if(ep > 0){
        x->x_epoc = ep;
    }
}

//--------------- topologia da rede (quantidade de neurônios vizinhos) ------------ //
static void som_topo (t_som *x, t_floatarg to){

    if(to <= MAXTOPO && to < x->x_neurons){
        x->x_topo = to;
    }
    else{
        error("the amount of neighbors allowed has exceeded or is higher than the number of neurons in the network");
    }

    gsl_vector_free (x->vizi);
    gsl_vector_free (x->index_vizi);
    x->vizi = gsl_vector_alloc (x->x_topo);
    x->index_vizi = gsl_vector_alloc (x->x_topo);
}

//----------------------- recebe raio de vizinhança --------------------------//
static void som_radius (t_som *x, t_floatarg ra){
  x->x_raiovizi = ra;
}

static void som_training (t_som *x, t_floatarg tra){
    x->x_trainingmode = tra;

    int trainingmode = x->x_trainingmode;

    switch(trainingmode) {
        case 0:
            post ("training mode: OFF");
            break;
        case 1: 
            post ("training mode: ON");
            break;
    }


}




//-------------------------- imprime as informações da rede ------------------------//
void som_print(t_som *x) {

    int i, j;
    int trainingmode = x->x_trainingmode;

    switch(trainingmode) {
        case 0:
            post ("training mode: OFF");
            break;
        case 1: 
            post ("training mode: ON");
            break;
    }

    post("number of neurons: %0.1f", x->x_neurons);
    post("vector size: %0.1f", x->x_vec);
    post("learning rate: %0.2f", x->x_learn);
    post("input size: %0.1f", x->x_input);
    post("current data: %0.1f", x->x_count);
    post("amount of epochs: %0.1f", x->x_epoc);
    post("current epoch: %0.1f", x->x_iter);
    post("topology: %0.1f", x->x_topo);
    post("neighbor radius: %0.2f", x->x_raiovizi);

   // Imprime os pesos dos neurônios
     for (i = 0; i < x->x_neurons; i++) {
         for (j = 0; j < x->x_vec; j++) {
            post("neuron(%d,%d): %g\n", i, j, gsl_matrix_get (x->m, i, j));
        } 
    }


}



//--------------------- inicializa os pesos dos neurônios de forma aleatória etre 0 e 1 (gsl) ---------------------//
void peso_init(t_som *x){
    int i, j;

    srand(time(NULL));

    for (i = 0; i < x->x_neurons; i++){
        for (j = 0; j < x->x_vec; j++){
            gsl_matrix_set (x->m, i, j, (double)rand()/RAND_MAX);
            // post("neuron(%d,%d): %g\n", i, j, gsl_matrix_get (x->m, i, j));
        }

    } 
    
 }


//------------------------- inicializa os pesos dos neurônios de forma linear entre 0 e 1 ----------------//

void peso_linear(t_som *x){
    float fator, startx, valuex, starty,valuey, total;
    int i;
    total = x->x_neurons + 0.0;
    startx = 0.0;
    starty = 1.0;
    fator = 1.0 / total;
    valuex = startx;
    valuey = starty;


    if (x->x_vec == 1){

        for (i = 0; i < x->x_neurons; i++){
                gsl_matrix_set (x->m, i, 0, valuex);
                valuex += fator;
        }
    }



    if (x->x_vec == 2){

        for (i = 0; i < x->x_neurons; i++){
                gsl_matrix_set (x->m, i, 0, valuex);
                gsl_matrix_set (x->m, i, 1, valuey);
                valuex += fator;
                valuey -= fator;
                
        }
     }

     if (x->x_vec == 3){

        for (i = 0; i < x->x_neurons; i++){
                gsl_matrix_set (x->m, i, 0, valuex);
                gsl_matrix_set (x->m, i, 1, valuey);
                gsl_matrix_set (x->m, i, 2, valuex);
                valuex += fator;
                valuey -= fator;
                
        }
     }

}
    

//-------------- reseta a rede ------------//

void som_reset (t_som *x){
     x->x_learn = 0.5;
     x->x_topo = 4;
     x->x_iter = 1;
     x->x_count = 0;
     x->x_raiovizi = 0.25;

     gsl_vector_free (x->vizi);
     gsl_vector_free (x->index_vizi);

     x->vizi = gsl_vector_alloc (x->x_topo);
     x->index_vizi = gsl_vector_alloc (x->x_topo);
     peso_init(x);

}



//------------------------ retorna os pesos dos neurônios no primeiro outlet quando recebe bang --------------------------//
void peso_bang(t_som *x){

    int i;
    int neuronx;

    neuronx = x->x_neurons;

    x->x_peso_outx[neuronx].a_type = A_FLOAT;
    x->x_peso_outy[neuronx].a_type = A_FLOAT;
    x->x_peso_outz[neuronx].a_type = A_FLOAT;


// se o vetor for unidimensional 

if (x->x_vec == 1){
    for(i = 0; i < x->x_neurons; i++){

        SETFLOAT (x->x_peso_outx + i, gsl_matrix_get (x->m, i, 0));    
        }

        outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
        
    }



//se o vetor for bidimensional

    if (x->x_vec == 2){
    for(i = 0; i < x->x_neurons; i++){

        SETFLOAT (x->x_peso_outx + i, gsl_matrix_get (x->m, i, 0));
        SETFLOAT (x->x_peso_outy + i, gsl_matrix_get (x->m, i, 1));    
        }

        outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
        outlet_anything(x->out1, gensym("y"), x->x_neurons, x->x_peso_outy);

    }


// se o vetor for tridimensional

if (x->x_vec == 3){
    for(i = 0; i < x->x_neurons; i++){
    
    SETFLOAT (x->x_peso_outx + i, gsl_matrix_get(x->m, i, 0));
    SETFLOAT (x->x_peso_outy + i, gsl_matrix_get(x->m, i, 1));
    SETFLOAT (x->x_peso_outz + i, gsl_matrix_get(x->m, i, 2));
    
    }

    outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
    outlet_anything(x->out1, gensym("y"), x->x_neurons, x->x_peso_outy);
    outlet_anything(x->out1, gensym("z"), x->x_neurons, x->x_peso_outz);
    }

}


static void trained_mode(t_som *x){
    int i, k, j, m;

    for (k = 0; k < x->x_neurons; k++) {
    x->x_d = 0.0;
        for (j = 0; j < x->x_vec; j++){
        gsl_vector_set (x->d, j, pow((gsl_matrix_get (x->m, k, j) - gsl_vector_get (x->data, j)), 2));
        x->x_d += gsl_vector_get(x->d, j);
        } 
        gsl_vector_set(x->dist, k, sqrt(x->x_d)); //distâncias entre o dado de entrada e todos os neurônios 
        //post("dist: %0.3f", gsl_vector_get(x->dist, k));
    }


//--------------------- ordena as distâncias euclidinanas do maior para o menor (testar com o quick sort) ------------------------//

    gsl_vector_memcpy(x->win, x->dist); //copia as distancias do array x_dist para o array x_win
    // gsl_sort_vector(x->win);




//-------------------------------- bubble sort (implementar com quick sort) -------------------------------------//
    
    for(i = 0; i < x->x_neurons; i++){
        for (k = 0; k < x->x_neurons-1; k++)

        if (gsl_vector_get (x->win, k) > gsl_vector_get (x->win, k+1)) {
            m = gsl_vector_get (x->win, k);
            gsl_vector_set (x->win, k, gsl_vector_get (x->win, k+1));
            gsl_vector_set (x->win, k+1, m);

        }
    }

        //imprime as distâncias em ordem crescente
    // for (i = 0; i < x->x_neurons; i++){
    //     post("dist ordenada: %0.3f", gsl_vector_get(x->win, i)); 

    //     }

// ---------------------------- encontra a posição da menor distância no array de distâncias x_dist ----------------------//
    
    int win_index = 0;
    win_index = gsl_vector_min_index(x->dist);



    int win;
    win = x->x_vec;

    x->x_peso_win[win].a_type = A_FLOAT;
    
    for(i = 0; i < x->x_vec; i++){
        SETFLOAT (x->x_peso_win + i, gsl_matrix_get (x->m, win_index, i));
        
    }

    x->x_index[1].a_type = A_FLOAT;

    SETFLOAT (x->x_index, win_index);

    outlet_anything(x->out1, gensym("winner"), x->x_vec, x->x_peso_win);
    outlet_anything(x->out1, gensym("index"), 1, x->x_index);



}



//----------------------- encontra neurônio vencedor e seus vizinhos, e atualiza seus pesos -----------------------//
void neuron_win(t_som *x, t_symbol *s, int argc, t_atom *argv) {

    int i;
    int j;
    int k;
    //int l;
    float m;

    
	
// // recebe lista de dados e armazena no vetor data usando gsl_vector    
	for (i = 0; i < argc; i++) {
        gsl_vector_set(x->data, i, argv[i].a_w.w_float); 
        } 

    int mode = x->x_trainingmode;

    switch (mode){
    case 0:
        trained_mode(x);
        break;
    case 1:

//------------------------ calcula distância euclidiana entre os dados de entrada e os pesos dos neurônios ----------------------//
 for (k = 0; k < x->x_neurons; k++) {
    x->x_d = 0.0;
        for (j = 0; j < x->x_vec; j++){
        gsl_vector_set (x->d, j, pow((gsl_matrix_get (x->m, k, j) - gsl_vector_get (x->data, j)), 2));
        x->x_d += gsl_vector_get(x->d, j);
        } 
        gsl_vector_set(x->dist, k, sqrt(x->x_d)); //distâncias entre o dado de entrada e todos os neurônios 
        //post("dist: %0.3f", gsl_vector_get(x->dist, k));
    }


//--------------------- ordena as distâncias euclidinanas do maior para o menor (testar com o quick sort) ------------------------//

    gsl_vector_memcpy(x->win, x->dist); //copia as distancias do array x_dist para o array x_win
    // gsl_sort_vector(x->win);




//-------------------------------- bubble sort (implementar com quick sort) -------------------------------------//
    
    for(i = 0; i < x->x_neurons; i++){
        for (k = 0; k < x->x_neurons-1; k++)

        if (gsl_vector_get (x->win, k) > gsl_vector_get (x->win, k+1)) {
            m = gsl_vector_get (x->win, k);
            gsl_vector_set (x->win, k, gsl_vector_get (x->win, k+1));
            gsl_vector_set (x->win, k+1, m);

        }
    }

        //imprime as distâncias em ordem crescente
    // for (i = 0; i < x->x_neurons; i++){
    //     post("dist ordenada: %0.3f", gsl_vector_get(x->win, i)); 

    //     }

// ---------------------------- encontra a posição da menor distância no array de distâncias x_dist ----------------------//

    int win_index = 0;
    win_index = gsl_vector_min_index(x->dist);
    // post("min dist index: %d", win_index);

    

//  //------------------------- atualiza os pesos do neurônio vencedor --------------------------//

    for(i = 0; i < x->x_vec; i++) {
        gsl_vector_set(x->wv, i, gsl_matrix_get(x->m, win_index, i) + x->x_learn * (gsl_vector_get(x->data, i) - gsl_matrix_get(x->m, win_index, i)));
        // post("peso update: %0.3f", gsl_vector_get(x->wv, i));
    }

    for(i = 0; i < x->x_vec; i++){
        gsl_matrix_set(x->m, win_index, i, gsl_vector_get(x->wv, i)); //atribui os pesos atualizados para o neurônio vencedor
    }
    


//-------------------- atualização dos neurônios vizinhos -----------------------//

// função de vizinhança
    float dif, vizifunc;


    
    for (i = 0; i < x->x_topo; i++){
        gsl_vector_set(x->vizi, i, gsl_vector_get(x->win, i+1)); 
        // post("dist vizi %0.3f", gsl_vector_get(x->vizi, i));
    }

//------------------------ encontra os índices dos neurônios vizinhos na matriz ---------------------//
    for (i = 0; i < x->x_topo; i++) {
        double vizi_value = gsl_vector_get(x->vizi, i);
        double min_diff = DBL_MAX;
        int min_index = -1;
    
        for (j = 0; j < x->x_neurons; j++) {
            double dist_value = gsl_vector_get(x->dist, j);
            double diff = fabs(vizi_value - dist_value);
            
            if (diff < min_diff) {
                min_diff = diff;
                min_index = j;
            
                }
            }
    
    if (min_index != -1) {
        gsl_vector_set(x->index_vizi, i, min_index);
        // post("vizi index: %d", min_index);
        dif = fabs(gsl_vector_get(x->vizi, i) - gsl_vector_get(x->win, 0)); //diferença absoluta entre a distância do neurônio vizinho e do neurônio vencedor
        vizifunc = exp(-dif / (2 * x->x_raiovizi * x->x_raiovizi)); //função de vizinhança
                for(k = 0; k < x->x_vec; k++){
                    gsl_matrix_set(x->m, min_index, k, gsl_matrix_get(x->m, min_index, k) + x->x_learn * vizifunc * (gsl_vector_get(x->data, k) - gsl_matrix_get(x->m, min_index, k))); //formula de atualização dos vizinhos
                    // post("vizi update: %0.3f", gsl_matrix_get(x->m, min_index, k));
                }   
            }
        }


//---------------------------- calcula o erro quadrático médio ---------------------------------//
    int itera = x->x_count;
    float error, z, y;
    z = 0;
    y = 0;
    int f;



    for (f = 0; f < x->x_vec; f++){ 

        gsl_vector_set(x->errodiff, f, gsl_vector_get(x->data, f) - gsl_vector_get(x->wv, f)); //diferença entre dado e peso atualizado
    }

    for(i = 0; i < x->x_neurons; i++){
        z += gsl_vector_get(x->errodiff, i);

    }


    gsl_vector_set(x->errosum, itera, z);
    
//----------------------- conta as iterações ---------------------//

    x->x_count++;
    if(x->x_count == x->x_input){
   
//----------------- atualiza a taxa de aprendizagem a partir de uma função exponecial decrescente -----------------------//

    x->x_learn = x->x_learn * exp(- x->x_iter / x->x_epoc);


//-----------------  atualiza o raio de vizinhança a partir de uma função exponencial decrescente

    x->x_raiovizi = x->x_raiovizi * exp(- x->x_iter / x->x_epoc);

//------------------ faz a soma dos quadrados das diferenças entre dados e neurônios vencedores atualizado -------------------- //


    for(i = 0; i< x->x_input; i++){
        y += gsl_vector_get(x->errosum, i);

    }

    error = y / (x->x_input * x->x_vec) * 1000; //erro médio quadrático

    }


//--------------- copia os pesos para um array de saída e envia para o outlet 1 ----------------//

    int neuronx;

    neuronx = x->x_neurons;


    x->x_peso_outx[neuronx].a_type = A_FLOAT;
    x->x_peso_outy[neuronx].a_type = A_FLOAT;
    x->x_peso_outz[neuronx].a_type = A_FLOAT;


// se o vetor for unidimensional 

if (x->x_vec == 1){
    for(i = 0; i < x->x_neurons; i++){

        SETFLOAT (x->x_peso_outx + i, gsl_matrix_get (x->m, i, 0));    
        }

        outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
        
    }

// se o vetor for bidimensional 
    if (x->x_vec == 2){
        for(i = 0; i < x->x_neurons; i++){
        
        SETFLOAT (x->x_peso_outx + i, gsl_matrix_get(x->m, i, 0));
        SETFLOAT (x->x_peso_outy + i, gsl_matrix_get(x->m, i, 1));
        
        }

        outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
        outlet_anything(x->out1, gensym("y"), x->x_neurons, x->x_peso_outy);
    }


// se o vetor for tridimensional

    if (x->x_vec == 3){
        for(i = 0; i < x->x_neurons; i++){
        
        SETFLOAT (x->x_peso_outx + i, gsl_matrix_get(x->m, i, 0));
        SETFLOAT (x->x_peso_outy + i, gsl_matrix_get(x->m, i, 1));
        SETFLOAT (x->x_peso_outz + i, gsl_matrix_get(x->m, i, 2));
        
        }

        outlet_anything(x->out1, gensym("x"), x->x_neurons, x->x_peso_outx);
        outlet_anything(x->out1, gensym("y"), x->x_neurons, x->x_peso_outy);
        outlet_anything(x->out1, gensym("z"), x->x_neurons, x->x_peso_outz);
    }





//--------------------- envia parametros atualizados para o segundo outlet à cada iteração -------------------//

    x->x_current_learn[1].a_type = A_FLOAT;
    x->x_current_iter[1].a_type = A_FLOAT;
    x->x_current_raiovizi[1].a_type = A_FLOAT;
    x->x_erro[1].a_type = A_FLOAT;

    SETFLOAT (x->x_current_learn, x->x_learn);
    SETFLOAT (x->x_current_iter, x->x_iter);
    SETFLOAT (x->x_current_raiovizi, x->x_raiovizi);
    SETFLOAT (x->x_erro, error);

    if(x->x_count == x->x_input){
        // y = 0.0;
        outlet_anything(x->out2, gensym("learning"), 1, x->x_current_learn);
        outlet_anything(x->out2, gensym("epoch"), 1, x->x_current_iter);
        outlet_anything(x->out2, gensym("radius"), 1, x->x_current_raiovizi);
        outlet_anything(x->out2, gensym("mse"), 1, x->x_erro);
        x->x_iter++;
        x->x_count = 0;

    }

//---------------------- envia um bang para o terceiro outlet quando atingir o numero maximo de iterações ----------------------//

    if(x->x_iter == x->x_epoc){
        outlet_bang(x->out3);
            }
    break;
    }
}


    


//--------------------------- salva os pesos dos neurônios em um arquivo de texto --------------------------------//

static void som_write(t_som *x, t_symbol *filename)
{
    FILE *fd;
    char buf[MAXPDSTRING];
    int i, j;
    canvas_makefilename(x->x_canvas, filename->s_name,
        buf, MAXPDSTRING);
    sys_bashfilename(buf, buf);
    if (!(fd = fopen(buf, "w")))
    {
        error("%s: can't create", buf);
        return;
    }
    for (i = 0; i < x->x_neurons; i++) {
        for (j = 0; j < x->x_vec; j++) {
            if (j > 0) {
                fprintf(fd, " ");
            }
        if (fprintf(fd, "%g", gsl_matrix_get(x->m, i, j)) < 0) {
            error("%s: write error", filename->s_name);
            goto fail;
        }
    }
    fprintf(fd, "\n");
}
fail:
    fclose(fd);
    post("file saved");
}

//------------------------------- lê os valores de um arquivo de texto e armazena na matriz ----------------------------//

static void som_read (t_som *x, t_symbol *filename, t_symbol *format){

    int filedesc, i, j, c, linha, coluna, prev_c;
    double value;

    FILE *fd;
    char buf[MAXPDSTRING], *bufptr;
    if ((filedesc = open_via_path(
        canvas_getdir(x->x_canvas)->s_name,
            filename->s_name, "", buf, &bufptr, MAXPDSTRING, 0)) < 0)
    {
        error("%s: can't open", filename->s_name);
        return;
    }
   
    fd = fdopen(filedesc, "r");

    linha = 0;
    coluna = 0;

    while((c = fgetc(fd)) != EOF) {
        if (c == '\n') {
            if(prev_c != '\n'){
                linha++;
            }
            coluna++;
        }
        else if (c == ' ' || c == '\t') {
            if(prev_c != ' ' && prev_c != '\t' && prev_c != '\n') {
            coluna++;
            }
        }
        prev_c = c;
    }
    if (prev_c != '\n'){
        linha++;
        coluna++;
    }
    coluna = coluna / linha;

    // atualiza o número de neurônios e de vetor com base no arquivo lido

    x->x_neurons = linha;
    x->x_vec = coluna;


    //desaloca memória
    gsl_matrix_free (x->m);
    gsl_vector_free (x->data);
    gsl_vector_free (x->d);
    gsl_vector_free (x->dist);
    gsl_vector_free (x->win);
    gsl_vector_free (x->wv);
    gsl_vector_free (x->errodiff);
    freebytes(x->x_peso_outx, x->x_neurons * sizeof(t_atom)); 
    freebytes(x->x_peso_outy, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_outz, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_win, x->x_vec * sizeof(t_atom));

    //aloca memórica com os novos valores
    x->m = gsl_matrix_alloc (x->x_neurons, x->x_vec);
    x->data = gsl_vector_alloc (x->x_vec);
    x->d = gsl_vector_alloc (x->x_vec);
    x->dist = gsl_vector_alloc (x->x_neurons);
    x->win = gsl_vector_alloc (x->x_neurons);
    x->wv = gsl_vector_alloc (x->x_vec);
    x->errodiff = gsl_vector_alloc (x->x_neurons);
    x->x_peso_outx = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom)); 
    x->x_peso_outy = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    x->x_peso_outz = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    x->x_peso_win = (t_atom *)getbytes(x->x_vec * sizeof(t_atom));

    fseek(fd, 0, SEEK_SET);



    for (i = 0; i < x->x_neurons; i++){
        for (j = 0; j < x->x_vec; j++) {
        if (fscanf(fd, "%lf", &value)!= 1)
            {
                error("%s: read error", filename->s_name);
                break;
            }
            gsl_matrix_set(x->m, i, j, value);
        }
    } 
    fclose(fd);
    post("file loaded with %0.1f neurons and %0.1f vectors", x->x_neurons, x->x_vec); 
} 

 

static void *som_new(t_symbol *s, int argc, t_atom *argv){ //argc é a quantidade de elementos da lista e argv é um ponteiro para uma lista
    t_som *x = (t_som *)pd_new(som_class);
     
     x->x_neurons = 10;
     x->x_vec = 2;
     x->x_learn = 0.5;
     x->x_topo = 4;
     x->x_iter = 1;
     x->x_count = 0;
     x->x_raiovizi = 0.25;
     x->x_input = 20;
     x->x_epoc = 100;
     x->m = gsl_matrix_alloc (x->x_neurons, x->x_vec);
     x->data = gsl_vector_alloc (x->x_vec);
     x->d = gsl_vector_alloc (x->x_vec);
     x->dist = gsl_vector_alloc (x->x_neurons);
     x->win = gsl_vector_alloc (x->x_neurons);
     x->wv = gsl_vector_alloc (x->x_vec);
     x->vizi = gsl_vector_alloc (x->x_topo);
     x->index_vizi = gsl_vector_alloc (x->x_topo);
     x->errodiff = gsl_vector_alloc (x->x_neurons);
     x->errosum = gsl_vector_alloc (x->x_input);
     x->x_peso_outx = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom)); 
     x->x_peso_outy = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
     x->x_peso_outz = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
     x->x_peso_win = (t_atom *)getbytes(x->x_vec * sizeof(t_atom));
     x->x_index = (t_atom *)getbytes(1 * sizeof(t_atom));
     x->x_current_learn = (t_atom *)getbytes(1 * sizeof(t_atom));
     x->x_current_iter = (t_atom *)getbytes(1 * sizeof(t_atom));
     x->x_current_raiovizi = (t_atom *)getbytes(1 * sizeof(t_atom));
     x->x_erro = (t_atom *)getbytes(1 * sizeof(t_atom)); 
     x->x_canvas = canvas_getcurrent();
     peso_init(x);
     x->out1 = outlet_new(&x->x_obj, &s_anything);
     x->out2 = outlet_new(&x->x_obj, &s_anything);
     x->out3 = outlet_new(&x->x_obj, &s_anything);
     return (x);
     
}


//função para destruir o objeto
void som_destroy(t_som *x){ 
	
    //freebytes(x->peso, sizeof(t_float));
    // freebytes(x->x_data, sizeof(t_float) * 2);
    // freebytes(x->x_dist, sizeof(t_float) * 4);
    // freebytes(x->x_arg, sizeof(t_float) * 4);
    gsl_matrix_free (x->m);
    gsl_vector_free (x->vizi);
    gsl_vector_free (x->index_vizi);
    gsl_vector_free (x->data);
    gsl_vector_free (x->d);
    gsl_vector_free (x->dist);
    gsl_vector_free (x->win);
    gsl_vector_free (x->wv);
    gsl_vector_free (x->errodiff);
    gsl_vector_free (x->errosum);
    freebytes(x->x_peso_outx, x->x_neurons * sizeof(t_atom)); 
    freebytes(x->x_peso_outy, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_outz, x->x_neurons * sizeof(t_atom));
    freebytes(x->x_peso_win, x->x_vec * sizeof(t_atom));
    freebytes(x->x_index, 1 * sizeof(t_atom));
    freebytes(x->x_current_learn, 1 * sizeof(t_atom));
    freebytes(x->x_current_iter, 1 * sizeof(t_atom));
    freebytes(x->x_current_raiovizi, 1 * sizeof(t_atom));
    freebytes(x->x_erro, 1 * sizeof(t_atom));
    outlet_free(x->out1); //desaloca memoria do outlet quando o objeto é destruido
    outlet_free(x->out2); //desaloca memoria do outlet quando o objeto é destruido
    outlet_free(x->out3);

}






//inicialização da classe - quando o objeto é carregado pelo pd essa função é ativada
void som_setup(void) {
	som_class = class_new(
		gensym("som"), //nome do objeto
		(t_newmethod)som_new, //chama a função construtor
		(t_method)som_destroy, //chama a função destruidor
		sizeof(t_som),
        CLASS_DEFAULT,
         A_DEFFLOAT, 0);//tamanho do objeto
		
    class_addlist(som_class, (t_method) neuron_win);
    class_addmethod(som_class, (t_method)peso_init, gensym("random"), A_GIMME, 0);
    class_addmethod(som_class, (t_method)peso_linear, gensym("linear"), A_GIMME, 0);
    class_addmethod(som_class, (t_method)som_training, gensym("training"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_print, gensym("print"), A_GIMME, 0);
    class_addmethod(som_class, (t_method)som_write, gensym("write"), A_SYMBOL, 0);
    class_addmethod(som_class, (t_method)som_read, gensym("read"), A_SYMBOL, 0);
    class_addbang(som_class, (t_method)peso_bang);
    class_addmethod(som_class, (t_method)som_neuron_size, gensym("netsize"), A_GIMME, 0);
    class_addmethod(som_class, (t_method)som_learn, gensym("learning"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_input, gensym("datasize"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_epoc, gensym("epochs"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_topo, gensym("topology"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_radius, gensym("nradius"), A_FLOAT, 0);
    class_addmethod(som_class, (t_method)som_reset, gensym("reset"), A_GIMME, 0);
}