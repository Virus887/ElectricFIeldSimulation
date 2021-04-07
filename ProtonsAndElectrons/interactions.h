#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 1200
#define H 800
#define TITLE_STRING "Damian Bis: Protons and electrons"




int proton_count = 5000;
int electron_count = 5000;
int big_proton_count = 1;
int big_electron_count = 1;


float DT = 0.005f;
float BIG_PARTICLE_POWER = 500000;

bool Paused = false;


float oldDT = 0.005f;
void keyboard(unsigned char key, int x, int y) {
	if (key == '+')DT += 0.001;
	if (key == '-')if (DT>=0.001)DT -= 0.001;

	if (key == 'w')BIG_PARTICLE_POWER += 10000;
	if (key == 's')BIG_PARTICLE_POWER -= 10000;

	if (key == 'r')
	{
		DT = 0.05f;
		BIG_PARTICLE_POWER = 500000;
	}
	if (key == 'p')
	{
		if (Paused)DT = oldDT;
		else
		{
			oldDT = DT;
			DT = 0.0;
		}
		Paused = !Paused;
	}
	if (key == 27)  exit(0);
	glutPostRedisplay();

}

void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_UP)BIG_PARTICLE_POWER += 10000;
	if (key == GLUT_KEY_DOWN)BIG_PARTICLE_POWER -= 10000;
	glutPostRedisplay();
}

void printInstructions() {
	printf("Instructions:\n");
	printf("To manipulate particles speed use '+' and '-' keys, \n");
	printf("To manipulate big particles power use 'up arrow' and 'down arrow' keys, alternative 'W' and 'S' \n");
	printf("To pause application use 'P' key. \n");
}
#endif