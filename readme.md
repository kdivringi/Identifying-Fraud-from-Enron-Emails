
#Identifying Fraud from Enron Email Dataset - UD120 Final Project
*By Kaan Divringi*




    import sys, os
    import pickle
    import pandas as pd
    import seaborn as sb
    %matplotlib inline
    
    sys.path.append("../tools/")
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to use the publicly available Enron dataset to predict corporate fraud. The company and events behind the dataset are a very public and notorious case in the public interest. The court proceedings as well as the large amount of emails released give an opportunity to have an intimate look at one of the most significant corporate fraud cases in history. Because the public records of the court give us exactly the people that were involved in corporate fraud, we can consider this problem a classification problem. In addition to the email corpus, from which some statistics such as from messages, to messages and messages involved with Persons Of Interest (I suppose hindsight is quite nice here!), we have some financial data such as salary, bonuses and stock values. We have continuous numeric values for supervised classification.

In the initial look at the data there were a few anomalies. Two of the rows are erroneous for the purposes of this project: “TOTAL” which is improperly taken from the legal document and “THE TRAVEL AGENCY IN THE PARK” which is not a person at all. Another row, “LOCKHART EUGENE E” does not have any features other than the label, negative for being a person of interest. Not particularly useful for our algorithm and he is removed (I noticed that the feature_format function does this automatically). Those are the samples, as for the features there are a few more anomalies that can be seen with a categorical box plot on the POI label.


    # The TOTAL entry is essentially an erroneous entry
    data_dict.pop('TOTAL')
    
    # EUGENE has no features at all, other than that he is not a POI
    ## NOTE: Looking at feature_format, it will also do this by default
    data_dict.pop('LOCKHART EUGENE E')
    
    # TRAVEL AGENCY IN THE PARK is indicative of shady dealing but not a person
    temp = data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
    
    def omit_NaN(d_dict):
    	"""Utility function for bringing into a Pandas DataFrame. Removing value
    	instead of having "NaN" allows pandas to assign the np.nan missing value"""
    	for name in d_dict.keys():
    		d = dict([(field, d_dict[name][field]) for field in d_dict[name].keys() 
    			if d_dict[name][field] != "NaN"])
    		d['name'] = name
    		yield(d)
    
    # Bring into Pandas DataFrame for easier exploration
    df = pd.DataFrame(omit_NaN(data_dict))
    numeric_cols = [col for col in df.columns if col not in ['name','email_address', 'poi']]
    for col in numeric_cols:
    	df[col] = df[col].astype('float32')
    
    # Boxplots vs POI plot of input vars
    n_cols = 4
    n_rows = len(numeric_cols) // n_cols
    if len(numeric_cols) % n_cols > 0:
    	n_rows += 1
    sb.plt.figure(figsize=(14,14))
    for i, col in enumerate(numeric_cols):	
    	sb.plt.subplot(n_rows, n_cols, i + 1)
    	sb.boxplot(df[col].dropna(), df.poi)
    #sb.set_context("notebook", font_scale=1)
    sb.plt.show()


![png](Final%20Project_files/Final%20Project_3_0.png)


As can be seen, there are two features (restricted_stock_deferred & director_fees) that do not have any observations in one of the two labels. These features also have quite low observation counts overall and another feature (loan_advances) only has 3 data points. These features are removed from consideration. 


    df2 = df.drop(["restricted_stock_deferred", "director_fees", "loan_advances"], axis=1)
    print "Total rows: %i" % len(df2)
    numeric_cols2 = [col for col in df2.columns if col not in ['name','email_address', 'poi']]
    print "Total columns: %i" % len(numeric_cols2)
    for col in numeric_cols2:
        print "%s non-null values: %i" % (col, sum(df2[col].notnull()))
    
    df3 = df2.drop("deferral_payments", axis=1)
    numeric_cols3 = [col for col in df3.columns if col not in ['name','email_address', 'poi']]

    Total rows: 143
    Total columns: 16
    bonus non-null values: 81
    deferral_payments non-null values: 38
    deferred_income non-null values: 48
    exercised_stock_options non-null values: 101
    expenses non-null values: 94
    from_messages non-null values: 86
    from_poi_to_this_person non-null values: 86
    from_this_person_to_poi non-null values: 86
    long_term_incentive non-null values: 65
    other non-null values: 91
    restricted_stock non-null values: 109
    salary non-null values: 94
    shared_receipt_with_poi non-null values: 86
    to_messages non-null values: 86
    total_payments non-null values: 123
    total_stock_value non-null values: 125
    

That leaves us with, 143 possible POIs overall, each with 16 numeric columns. For several of the features there are quite a few missing values, and deferral_payments has the most of the remaining features. After that feature is removed, all of the remaining features have more than a 1/3 of all the values filled. Most have at least half. At this point I felt it premature to remove more features without a good reason. I anticipated using feature selection and dimensionality reduction to remove features from here.

At this point it would be prudent to check the distribution of our label accross the remaining features:



    poi_df = df3.groupby('poi', as_index=False).get_group(True)
    
    for col in numeric_cols3:
        print "%s: %i POI / %i total non-null" % (col, sum(poi_df[col].notnull()), sum(df3[col].notnull()))

    bonus: 16 POI / 81 total non-null
    deferred_income: 11 POI / 48 total non-null
    exercised_stock_options: 12 POI / 101 total non-null
    expenses: 18 POI / 94 total non-null
    from_messages: 14 POI / 86 total non-null
    from_poi_to_this_person: 14 POI / 86 total non-null
    from_this_person_to_poi: 14 POI / 86 total non-null
    long_term_incentive: 12 POI / 65 total non-null
    other: 18 POI / 91 total non-null
    restricted_stock: 17 POI / 109 total non-null
    salary: 17 POI / 94 total non-null
    shared_receipt_with_poi: 14 POI / 86 total non-null
    to_messages: 14 POI / 86 total non-null
    total_payments: 18 POI / 123 total non-null
    total_stock_value: 18 POI / 125 total non-null
    

It looks like there are a decent number of POIs in each of the remaining features. The POI to non-POI ratio is definitely not evenly distributed, though. Only about 10-20% of the total observations in each feature are POIs. Any future train-test splits will need to take this into account in order to have a relevant validation method.

A look at a pair plot colored by the POI gives a very high bird’s eye view of the situation. There are not really much in the way of obvious clusters or distributions but we see some values that can be considered outliers. These outliers are grounded in reality, however, as far as I can tell. While it is possible to have an opinion on CEO vs worker pay, there is nothing technically wrong with the large values and nothing needs to be addressed, other than some scaling as required by the algorithm when it comes to that. That will be taken care of in the pipeline process.


    least_null_cols = ["exercised_stock_options", "restricted_stock", "bonus", "expenses", "from_messages",
                       "from_poi_to_this_person", "from_this_person_to_poi", "other", "salary",
                       "shared_receipt_with_poi", "to_messages", "total_payments", "total_stock_value", "poi"]
    sb.pairplot(df3[least_null_cols].dropna(), hue="poi")




    <seaborn.axisgrid.PairGrid at 0x1940c128>




![png](Final%20Project_files/Final%20Project_9_1.png)


>What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]


I did attempt to engineer a feature, using some techniques that were briefly touched on in class. We have quite a bit of text data that seemed to be unused in the full email corpus (currently, metadata such as who sent/received emails is the only way the email_corpus is included in the features) so I set out to apply the text learning skills that were taught in the class. I was able to find a list of emails from each person in the emails_by_address directory. Using this procedure I was able to automatically read in and associate the emails by person in the dataset in a format suitable for vectorization. I also stemmed and removed any word with digits (there was a lot of email shorthand with numbers in them that were the rarest words but also gibberish). I then fed that into the Inverse Document Frequency Vectorizer, with the English stopwords defined and a maximum frequency of 0.5. I also read every 10th email for the sake of time and efficiency. This led me to have 56k words with their idf frequencies. Some of the infrequent words are shown below:


    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import RandomizedPCA, TruncatedSVD
    
    t = pickle.load(open('email_text.pkl', 'r'))
    vect = TfidfVectorizer(sublinear_tf=True, max_df = 0.5, stop_words="english")
    freqs = vect.fit_transform(t)
    
    print "# of features: %i" % len(vect.idf_)
    pd.np.random.seed(402)
    pd.np.random.choice(vect.get_feature_names(), size=(100))

    # of features: 56030
    




    array([u'adapt', u'billionair', u'tricon', u'transactionsday', u'nvert',
           u'yeunghou', u'baxterect', u'hamiltonenronenronxg', u'asociado',
           u'roberttetzdgscagov', u'recentl', u'greenwich',
           u'eugenegaycpmxsaiccom', u'philnailsasmcagov', u'kenagi', u'gabi',
           u'arcordia', u'ciscon', u'cadden', u'electricfrom', u'mrs',
           u'tyumen', u'interconnectionintegr', u'queu',
           u'mjhamiltbellatlanticnet', u'hunaid', u'iabc', u'mozilla', u'doak',
           u'rdonenronenronxg', u'hirsch', u'stri', u'cession', u'saynoth',
           u'tregareuenronenron', u'semperg', u'analystassociatesadmin',
           u'yorkshir', u'pmet', u'huntingtonerinblillycom',
           u'cbsmarketwatchcom', u'jerald', u'electricitygener', u'fino',
           u'appendic', u'scholl', u'achiev', u'sixfold', u'winbut',
           u'harebrain', u'blister', u'kishkillenrondevelopmentenrondevelop',
           u'bortolottilonectect', u'christopherwalkerlinklaterscom',
           u'mississipp', u'impac', u'communicationmeet', u'rve', u'vmail',
           u'gaillardeuenron', u'pamela', u'itectur', u'berkeley',
           u'importdepend', u'triniti', u'sbr', u'proposalsmani', u'henhous',
           u'sevenweek', u'sweeter', u'gbisti', u'klauer', u'pressu',
           u'lightsout', u'eclip', u'candic', u'investco', u'trajectori',
           u'leino', u'exwif', u'word', u'canaletto', u'reengin', u'proenron',
           u'lipper', u'peytonhouect', u'guest', u'breachofcontract',
           u'leetwinstimcom', u'wangfariceedu', u'taxhelp', u'santiko',
           u'leasteffici', u'privatelyown', u'msheehanadventiscom', u'thought',
           u'salari', u'generationpow', u'phamaceut', u'mngt'], 
          dtype='<U70')



I judged this a rather unwieldly addition to the dataset so rather than merging this directly onto the feature list, I did an unsupervised clustering algorithm to reduce the amount of new features and possibly discover new and more powerful metafeatures. I used the TruncatedSVD due to a depreciation warning but we have not covered this in class. I tried this for a number of different # of components and I found that with 50 components I could explain about 80% of the variation and I went with that. 


    # Run a little parameter study on number of components vs explained var ratio
    comps = [2, 5, 10, 25, 50, 75, 100, 150]
    evar_rat =[]
    for num in comps:
    	pca = TruncatedSVD(n_components=num)
    	pca.fit(freqs)
    	evar_rat.append(sum(pca.explained_variance_ratio_))
    
    sb.set_style("whitegrid")
    sb.plt.plot(comps,evar_rat)
    sb.plt.xlabel("# Principle Components")
    sb.plt.ylabel("Explained Variance Ratio")
    sb.plt.show()


![png](Final%20Project_files/Final%20Project_13_0.png)


So instead of 56k features we have 50, an efficiency savings of about 3 orders of magnitude. Looking back, however, I do have some regrets and I may do things differently in the future. For one thing, performance was a key variable in employing the dimensionality reduction but my best algorithm turned out to be Naïve Bayes, something which is very performant and well suited to this type of work anyways. If I had known this in advance, I would have kept more of the 56k features. The other issue is that despite all of the work that I did in incorporating this new type of feature, it did not end up improving the algorithm, as will be shown later. I did spend some time on things like FeatureUnions in trying to make a hyper parameter of the n-gram length and document frequency but due to a shortage of both computer time and engineering time I was not able to. It was a fair amount of programming work for not a lot of payoff, and that may be a lesson in and of itself.

My final algorithm ending up using just 5 features to fulfil the requirements of the assignment:
The SelectKBest algorithm automatically selected the following 5 features: salary, exercised_stock_options, bonus, restricted_stock and total_stock_value. These items were scaled with the standard scalar before and classified with a Gaussian Naïve Bayes algorithm. I don’t believe that scaling is strictly required for the Naive Bayes Algorithm but I felt the StandardScalar might express the magnitude of some of the outliers in features. The Standard vs MinMax do not make a difference in the results, however.



    from sklearn.grid_search import GridSearchCV
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import StratifiedShuffleSplit
    from tester import load_classifier_and_data
    from feature_format import featureFormat, targetFeatureSplit
    
    # Load Features, Labels of dataset to use
    clf, dataset, feature_list = load_classifier_and_data()
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    # Make the pipeline
    clf = make_pipeline(StandardScaler(), SelectKBest(), GaussianNB())
    
    # Make the parameter grid
    params = [
        {'selectkbest__k':range(1,len(feature_list))}
        ]
    
    # Create cross validator for recall, since that gave the most trouble
    cv = StratifiedShuffleSplit(labels, n_iter = 200, random_state = 142)
    
    # Do the grid search, line plot
    grid_search = GridSearchCV(clf, params, cv = cv, scoring = 'recall')
    grid_search.fit(features, labels)
    x = [item.parameters['selectkbest__k'] for item in grid_search.grid_scores_]
    y = [item.mean_validation_score for item in grid_search.grid_scores_]
    
    grid_search2 = GridSearchCV(clf, params, cv = cv, scoring = 'precision')
    grid_search2.fit(features, labels)
    y2 = [item.mean_validation_score for item in grid_search2.grid_scores_]
    
    sb.plt.plot(x, y, label="Recall")
    sb.plt.plot(x, y2, label="Precision")
    sb.plt.xlabel("K (# of features)")
    sb.plt.ylabel("Mean Validation Metric")
    sb.plt.title("Validation Metrics w/o Engineered Text Features")
    sb.plt.legend()




    <matplotlib.legend.Legend at 0x1aa432b0>




![png](Final%20Project_files/Final%20Project_15_1.png)


The above show the validation curves for the feature set without the engineered text features. The sweet spot here is at 5 features. Compare this to the curve with the engineered features.


    # This code snippet is run with pkl files that include engineered features
    clf, dataset, feature_list = load_classifier_and_data()
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    # Make the pipeline
    clf = make_pipeline(StandardScaler(), SelectKBest(), GaussianNB())
    
    # Make the parameter grid
    params = [
        {'selectkbest__k':range(1,len(feature_list))}
        ]
    
    grid_search = GridSearchCV(clf, params, cv = cv, scoring = 'recall')
    grid_search.fit(features, labels)
    x = [item.parameters['selectkbest__k'] for item in grid_search.grid_scores_]
    y = [item.mean_validation_score for item in grid_search.grid_scores_]
    
    grid_search2 = GridSearchCV(clf, params, cv = cv, scoring = 'precision')
    grid_search2.fit(features, labels)
    y2 = [item.mean_validation_score for item in grid_search2.grid_scores_]
    
    sb.plt.plot(x, y, label="Recall")
    sb.plt.plot(x, y2, label="Precision")
    sb.plt.xlabel("K (# of features)")
    sb.plt.ylabel("Mean Validation Metric")
    sb.plt.title("Validation Metrics WITH the Engineered Text Features")
    sb.plt.legend()




    <matplotlib.legend.Legend at 0x1a787cc0>




![png](Final%20Project_files/Final%20Project_17_1.png)


The recall and precision are not above the threshold at the same time here. The rest of the investigation proceeds without the engineered features.

The feature scores as selected by the K Best Algorithm with the chi^2 test are as follows:


    k_scores = zip(feature_list[1:], grid_search.best_estimator_.steps[1][1].scores_)
    k_scores = sorted(k_scores, key = lambda k: k[1], reverse = True)
    print pd.DataFrame(k_scores, columns = ["feature", "chi_square"])

                        feature  chi_square
    0   exercised_stock_options   24.815080
    1         total_stock_value   24.182899
    2                     bonus   20.792252
    3                    salary   18.289684
    4           deferred_income   11.458477
    5       long_term_incentive    9.922186
    6          restricted_stock    9.212811
    7            total_payments    8.772778
    8   shared_receipt_with_poi    8.589421
    9                  expenses    6.094173
    10  from_poi_to_this_person    5.243450
    11                    other    4.187478
    12  from_this_person_to_poi    2.382612
    13              to_messages    1.646341
    14            from_messages    0.169701
    

>What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

The best algorithm in terms of metric scores and performance was the Gaussian Naïve Bayes algorithm. It was the only algorithm that I was able to get to fulfil the performance requirements of the project. I compared it with two algorithms:

Logistic Regression: I found it highly recommended from a source on Quora. It was a fairly quick algorithm, though not as quick as Naïve Bayes. It generally was able to achieve high precision scores but could not clear the bar for Recall. This algorithm was scaled in the pipeline and the KBestFeature selection was also used, although I had been lead to believe that it should eliminate features on it’s own. It is not an algorithm that we covered in class and I will have to study it further.

Adaboost: I wanted to use an ensemble algorithm to round out the algorithm types that I employed in this project. The Adaboost algorithm didn’t need to be scaled or features selected due to how it works internally. It was the most computationally expensive algorithm by far of the ones that I tried but it did come closer than LogisticRegression to meeting the project requirements.


|    Accuracy    |    Precision    |    Recall    |    Notes                                               |
|----------------|-----------------|--------------|--------------------------------------------------------|
|    0.84987     |    0.42232      |    0.3425    |    With txt features NaiveBayes best config            |
|    0.86213     |    0.45913      |    0.191     |    With txt features LogisticRegression best config    |
|    0.84027     |    0.37093      |    0.2845    |    With txt features Adaboost best config              |
|    0.84833     |    0.41964      |    0.359     |    w/o txt features NaiveBayes best config             |
|    0.86253     |    0.45765      |    0.1675    |    w/o txt features LogisticRegression best config     |
|    0.84413     |    0.37859      |    0.2635    |    w/o txt features Adaboost best config               |

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

The major algorithms are all quite powerful but they can also be fragile due to how they are applied to general purpose problems. It becomes necessary to tune certain general parameters for some algorithms and for some problems in order to get the optimal algorithm performance. For the final algorithm that I used, Naïve Bayes, it did not actually require parameter tuning. The feature selection step below did, however. For this, I attempted to tune the target number of features through the GridSearchCV method. One difficulty in this was that the GridSearch seems to be based off of either an accuracy score or nothing at all. I would run the grid search on a parameter of interest and the selected algorithm would perform worse than with the defaults. I could not figure out how to hook cross validation into the gridsearch algorithm so I essentially did so manually using the tester code, which uses a StratifiedShuffle cross validation to evaluate the algorithm. For the LogisticRegression I tuned C as well as number of features, k. For the AdaBoost algorithm I tuned learning speed and number of estimators.

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validation is more than just error checking, it is checking to make sure that the model applies in the real world. The classic mistake is overfitting and this is caused by not holding back some of the data for testing purposes. It is possible to make a model work very well on a training data but to have it be pretty poor in anything new. My model was validated by a StratifiedShuffle Algorithm that splits the data while still keeping the same proportion of labels in each split. This is important because there are a relatively low number of observations and an even lower amount of POIs, positive labels. 

>Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The primary usage validation metrics in this project were precision and recall. In general, precision was higher across all of the algorithms used, with some of them reaching above 0.4 without too much trouble. Recall was generally the most difficult, and the average of this across all three algorithms used was 0.26. This means that the algorithm was not as good at identifying someone as a POI given that the person is a POI to begin with. On the other hand it is not as likely to misidentify a non-POI as a POI. I would say that this algorithm is more likely to give a false negative than a false positive.

The lesser amount of difficulty with precision means that it was less likely to falsely identify somebody as  POI when they do not fit that label. Essentially, if someone is not a POI, this algorithm is not as likely to identify them as one. This metric, however, may be helped by the fact that there are relatively few POIs in the dataset. It is unlikely that the precision could get very low without the accuracy being massively affected so I anticipate it being difficult to even train such an algorithm.
