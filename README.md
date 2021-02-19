# Requirements
To run this project, make sure that `Python 3.7` and `jupyter` are installed. 

You will need to install certain python libraries. To do so, open a terminal, and execute (from this directory):

```pip install -r requirements.txt```

If you are using Anaconda, run instead (from this directory, with the appropriate Conda environment activated):

```conda install --file requirements.txt```

# Running the project
To run the project, start a Jupyter Notebook instance (`jupyter notebook` in a terminal) and open the Notebook in this project.

From there, `Cell > Run All` will run all the cells. After a succesful execution of the project, three files will be created, each corresponding to a model (Naive Bayes, Base Decision Tree, Best Decision Tree), containing the results from evaluating a model on a portion of the data set. By default, the data set is the `all_sentiment_shuffled.txt` file contained in the same directory, with 80% of the data set used to train the models, and 20% of the data set used to evaluate the models.

# Team Members
Team \#13

Luc Nguyen (40097582)

Chelsea Guan (40097861)

Joseph Loiselle (40095345)

Mounceph Morssaoui (40097557)