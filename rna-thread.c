#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <pthread.h>

/*
    ce : couche d'entree
    cc : couche cachee
    cs : couche sortie
    w : poids
    b : biais
*/

// taux d'apprentissage
static double TAUX_APPRENTISSAGE = 0.1f;
// nombre d'iteration de l'apprentissage
static int EPOCHS = 10000;



typedef struct {
    double *valeur_poids_biais; // tableau avec les valeurs des poids/ biais (w & b)
    double *valeur_neurone; // tableau avec les valeurs des neuronnes
    double *valeur_erreur_neurone; // erreur neurones
    double *neurone_entree; // liste des val ce
    double *neurone_sortie; //liste des val cs
    double *valeur_erreur_sortie; // erreur de cs
    int *index_couche; // indice de debut des couches (exemple [0(ce),2(cc),4(cs)])
    int *taille_couche; // taille des couches (exemple [2(ce),2(cc),1(cs)])
    int *index_poids_biais; // indice des poids/ biais (exemple [0(ce),5(cc),3(cs)])
    int nombre_couche; // nombre de couche total (XOR = 3 ici)
    int nombre_neurone; // nombre total de noeurones n
    int nombre_poids_biais; // w+biais
    int taille_couche_entree; // taille ce
    int taille_couche_sortie; // taille cs
} mlp;

typedef struct {
	mlp *reseau; // propagation, retropropagation
	double *resultat_attendu;

} thread_args;




double generate_rand() {
    double x = (double)rand()/(double)(RAND_MAX);
    return x;
}

double sigmoid(double x) {
     double result;
     result = 1 / (1 + exp(-x));
     return result;
}

double sigmoid_prime(double x) {
     double result;
     result = 1.0 - x * x;
     return result;
}

/*
    - initialisation du mlp
    - allocation memoire
    - generer poids & biais

*/
mlp *init(int nombre_couche, int *taille_couche) {

    mlp *reseau = malloc(sizeof * reseau);

    reseau->nombre_couche = nombre_couche;
    reseau->taille_couche = malloc(sizeof * reseau->taille_couche * reseau->nombre_couche);
    reseau->index_couche = malloc(sizeof * reseau->index_couche * reseau->nombre_couche);

    int i;
    reseau->nombre_neurone = 0;
    for (i = 0; i < nombre_couche; i++) {
        reseau->taille_couche[i] = taille_couche[i];
        reseau->index_couche[i] = reseau->nombre_neurone;
        reseau->nombre_neurone += taille_couche[i];
    }

    reseau->valeur_neurone = malloc(sizeof * reseau->valeur_neurone * reseau->nombre_neurone);
    reseau->valeur_erreur_neurone = malloc(sizeof * reseau->valeur_erreur_neurone * reseau->nombre_neurone);

    reseau->taille_couche_entree = taille_couche[0];
    reseau->taille_couche_sortie = taille_couche[nombre_couche-1];
    reseau->neurone_entree = reseau->valeur_neurone;
    reseau->neurone_sortie = &reseau->valeur_neurone[reseau->index_couche[nombre_couche-1]];
    reseau->valeur_erreur_sortie = &reseau->valeur_erreur_neurone[reseau->index_couche[nombre_couche-1]];

    reseau->index_poids_biais = malloc(sizeof * reseau->index_poids_biais * (reseau->nombre_couche-1));


    reseau->nombre_poids_biais = 0;
    for (i = 0; i < nombre_couche - 1; i++) {
        reseau->index_poids_biais[i] = reseau->nombre_poids_biais;
        // poids & biais
        reseau->nombre_poids_biais += (reseau->taille_couche[i]+1) * reseau->taille_couche[i+1];
    }

    reseau->valeur_poids_biais = malloc(sizeof * reseau->valeur_poids_biais * reseau->nombre_poids_biais);

    for (i = 0; i < reseau->nombre_poids_biais; i++) {
        reseau->valeur_poids_biais[i] = 0.5 * pow(generate_rand(),2);
    }
    return reseau;
}


void set_entrees(mlp * reseau, double *entrees) {
    int i;
    for (i = 0; i < reseau->taille_couche_entree; i++) {
      reseau->neurone_entree[i] = entrees[i];
    }
}

/*
    - entrées
    - propagation avec la fonction d'activation sigmoid
*/
void propagation (mlp * reseau) {

    int index_poids_biais; int i;
    index_poids_biais = 0;
    // parcours toutes les couches sauf entree
    for (i = 1; i < reseau->nombre_couche; i++) {
        int j;
        // parcours la couche courante grace a la table des indices
        for (j = reseau->index_couche[i]; j < reseau->index_couche[i] + reseau->taille_couche[i]; j++) {
            double somme_ponderee = 0.0;
            int k;
            // on parcours les neurones de la couche precedente
            for (k = reseau->index_couche[i-1]; k < reseau->index_couche[i-1] + reseau->taille_couche[i-1]; k++) {
                somme_ponderee += reseau->valeur_neurone[k] * reseau->valeur_poids_biais[index_poids_biais];
                index_poids_biais++;
            }
            // biais
            somme_ponderee += reseau->valeur_poids_biais[index_poids_biais];
            index_poids_biais++;

            reseau->valeur_neurone[j] = somme_ponderee;
            // fonction de propagation sur la cc
            //if (i != reseau->nombre_couche - 1) reseau->valeur_neurone[j] = 1.0 / (exp(-somme_ponderee) + 1.0);
            if (i != reseau->nombre_couche - 1) reseau->valeur_neurone[j] = sigmoid(somme_ponderee);
        }
    }
}



/*
    - apprentissage
    - calcul erreur
    - retropropagation
    - maj des poids

*/
void retropropagation(mlp *reseau, double *resultat_attendu) {

    int i;
    // premier poids/ biais cc
    int index_poids_biais = reseau->index_poids_biais[reseau->nombre_couche-2];

    /*
        - retropropagation de l'erreur
    */

   // parcours neuronnes sorties
    for (i = 0; i < reseau->taille_couche_sortie; i++) {
        // calcul erreur du neuronne courant
        reseau->valeur_erreur_sortie[i] = reseau->neurone_sortie[i] - resultat_attendu[i];

        int j;
        // parcours tout les neuronnes de la couche cachee
        for (j = reseau->index_couche[reseau->nombre_couche-2]; j < reseau->index_couche[reseau->nombre_couche-2] + reseau->taille_couche[reseau->nombre_couche-2]; j++) {
            // maj poids w avec l'erreur de sortie
            // * taux d'apprentissage sinon les valeur s'envolent : nan
            reseau->valeur_poids_biais[index_poids_biais] -= TAUX_APPRENTISSAGE * reseau->valeur_erreur_sortie[i] * reseau->valeur_neurone[j];
            index_poids_biais++;
        }
        // maj des biais de la cc courante
        reseau->valeur_poids_biais[index_poids_biais] -= reseau->valeur_erreur_sortie[i];
        index_poids_biais++;
    }

    /*
        - retropropagation avec la fonction dérivée
        - parcourir le mlp a l'envers
        - maj erreur neuronnes avec les erreurs de la couche suivante
        - maj poids avec : - fonction derivee(n+1)*erreur(n+1)*neuronne(n+2)
        - maj biais : - fonction derivee(n+1)*erreur(n+1)
    */

    // on parcours les couche cachee dans le sens inverse en excluant la couche d'entrée
    for (i = reseau->nombre_couche - 2; i > 0; i--) {
        int j;
        int jj= 0;
        // indice debut poids/ biais de la couche precedent (cachee ou sortie)
        int index_poids_biais = reseau->index_poids_biais[i-1];

        // parcours tous les neuronnes de la couche cachee courant
        for (j = reseau->index_couche[i]; j < reseau->index_couche[i] + reseau->taille_couche[i]; j++,jj++) {
            int k;
            // on se positionne sur la poid qui relie la cc courant à un neurone de la couche suivante
            int index_poids_biais2 = reseau->index_poids_biais[i] + jj;

            // on somme les valeurs d'erreur des neurones de la couche suivante
            for (k = reseau->index_couche[i+1]; k < reseau->index_couche[i+1] + reseau->taille_couche[i+1]; k++) {
                // init erreur neuronne courant
                reseau->valeur_erreur_neurone[j] = 0;
                reseau->valeur_erreur_neurone[j] += reseau->valeur_poids_biais[index_poids_biais2] * reseau->valeur_erreur_neurone[k];
                index_poids_biais2+=reseau->taille_couche[i]+1;
            }

            // pour tous les neuronnes de la cc precedente
            for (k = reseau->index_couche[i-1]; k < reseau->index_couche[i-1] + reseau->taille_couche[i-1]; k++) {

                // sigmoid prime poids (fonction derivee)
                reseau->valeur_poids_biais[index_poids_biais] -= sigmoid_prime(reseau->valeur_neurone[j]) * reseau->valeur_erreur_neurone[j] * TAUX_APPRENTISSAGE * reseau->valeur_neurone[k];
                index_poids_biais++;
            }
            // sigmoid prime biais
            reseau->valeur_poids_biais[index_poids_biais] -= sigmoid_prime(reseau->valeur_neurone[j]) * reseau->valeur_erreur_neurone[j] * TAUX_APPRENTISSAGE;
            index_poids_biais++;
        }
    }
}



	/*
		Apprentissage avec XOR & thread

	*/

	void *runing_xor(thread_args*arguments) {

    // dataset
    double xor[4][3] = {
        // A | B | A XOR B
        {-1,-1,  -1},
        {-1, 1,   1},
        {1, -1,   1},
        {1,  1,  -1}
    };



    int i;pthread_t thread1;
    int iret1; int iret2;



    // apprentissage

    for (i=0; i < EPOCHS; i++) {
        // distribution aléatoire du dataset
        int tuple_rand = 4.0*rand()/(RAND_MAX+1.0);
        set_entrees(arguments->reseau,&xor[tuple_rand][0]);

        arguments->resultat_attendu = &xor[tuple_rand][2];

	propagation(arguments->reseau);
	retropropagation(arguments->reseau,arguments->resultat_attendu);

	}


	pthread_exit(NULL);
	}




	/*
		Apprentissage avec AND & thread

	*/

	void *runing_and(thread_args*arguments) {

    // dataset

            double and[4][3] = {
        {-1,-1,  -1},
        {-1, 1,   -1},
        {1, -1,   -1},
        {1,  1,  1}
    };

    int i;pthread_t thread1;
    int iret1; int iret2;



    // apprentissage
    for (i=0; i < EPOCHS; i++) {
        // distribution aléatoire du dataset
        int tuple_rand = 4.0*rand()/(RAND_MAX+1.0);
        set_entrees(arguments->reseau,&and[tuple_rand][0]);
        arguments->resultat_attendu = &and[tuple_rand][2];

	propagation(arguments->reseau);
	retropropagation(arguments->reseau,arguments->resultat_attendu);

	}


	pthread_exit(NULL);
	}



int main() {

    srand(time(NULL));

    static pthread_mutex_t mutex_thread = PTHREAD_MUTEX_INITIALIZER;


    // init
    int taille_couche[] = {2, 4, 1};
    mlp * reseau = init(3, taille_couche);


    // dataset
    double xor[4][3] = {
        // A | B | A XOR B
        {-1,-1,  -1},
        {-1, 1,   1},
        {1, -1,   1},
        {1,  1,  -1}
    };

        double and[4][3] = {
        {-1,-1,  -1},
        {-1, 1,   -1},
        {1, -1,   -1},
        {1,  1,  1}
    };

    int i;
    pthread_t thread1;
    pthread_t thread2;
    int iret1; int iret2;




//        thread_args thread1_args = init_thread(reseau,&xor[tuple_rand][2]);
	thread_args *thread1_args = malloc(sizeof * thread1_args);
	thread1_args->reseau = reseau;
	thread1_args->resultat_attendu = NULL;

        pthread_mutex_lock(&mutex_thread);
        iret1 = pthread_create(&thread1, NULL, runing_xor, (void*)thread1_args);
//        iret1 = pthread_create(&thread1, NULL, set_entrees, (void *) input);

    	pthread_mutex_unlock(&mutex_thread);





    	pthread_join(thread1, NULL);
//    	    	pthread_join(thread2, NULL);

//        propagation(reseau);
//        retropropagation(reseau,&xor[tuple_rand][2]);



   // pthread_join(thread1, &thread_result);

    // test
    printf("XOR\n");
    printf(" A | \tB | \tATTENDU | \tOBTENU PAR APPRENTISSAGE\n");
    set_entrees(reseau,&xor[0][0]);
    propagation(reseau);
	printf(" %f\t %f\t %f\t %f\n", xor[0][0], xor[0][1], xor[0][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[1][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", xor[1][0], xor[1][1], xor[1][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[2][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", xor[2][0], xor[2][1], xor[2][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[3][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", xor[3][0], xor[3][1], xor[3][2], reseau->neurone_sortie[0]);


    printf(" %d", xor[3][0], xor[3][1], xor[3][2], reseau->neurone_sortie[0]);



    //        thread_args thread1_args = init_thread(reseau,&xor[tuple_rand][2]);
	thread_args *thread2_args = malloc(sizeof * thread2_args);
	thread2_args->reseau = reseau;
	thread2_args->resultat_attendu = NULL;

   //     pthread_mutex_lock(&mutex_thread);

        iret2 = pthread_create(&thread2, NULL, runing_and, (void*)thread2_args);
//        iret2 = pthread_create(&thread2, NULL, set_entrees, (void *) input);

    //	pthread_mutex_unlock(&mutex_thread);

    	pthread_join(thread2, NULL);

    // test
    printf("AND\n");
    printf(" A | \tB | \tATTENDU | \tOBTENU PAR APPRENTISSAGE\n");
    set_entrees(reseau,&xor[0][0]);
    propagation(reseau);
	printf(" %f\t %f\t %f\t %f\n", and[0][0], and[0][1], and[0][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[1][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", and[1][0], and[1][1], and[1][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[2][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", and[2][0], and[2][1], and[2][2], reseau->neurone_sortie[0]);
    set_entrees(reseau,&xor[3][0]);
    propagation(reseau);
    printf(" %f\t %f\t %f\t %f\n", and[3][0], and[3][1], and[3][2], reseau->neurone_sortie[0]);




    return 0;
}
