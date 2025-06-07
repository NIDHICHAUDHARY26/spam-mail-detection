.. code:: ipython3

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score

.. code:: ipython3

    data = pd.read_csv(r'C:\Users\agri2024\Downloads\PP6\spam.csv', encoding='ISO-8859-1')
    data.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>v1</th>
          <th>v2</th>
          <th>Unnamed: 2</th>
          <th>Unnamed: 3</th>
          <th>Unnamed: 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ham</td>
          <td>Go until jurong point, crazy.. Available only ...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ham</td>
          <td>Ok lar... Joking wif u oni...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>spam</td>
          <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ham</td>
          <td>U dun say so early hor... U c already then say...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ham</td>
          <td>Nah I don't think he goes to usf, he lives aro...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    data = data[['v1', 'v2']]
    data.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>v1</th>
          <th>v2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ham</td>
          <td>Go until jurong point, crazy.. Available only ...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ham</td>
          <td>Ok lar... Joking wif u oni...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>spam</td>
          <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ham</td>
          <td>U dun say so early hor... U c already then say...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ham</td>
          <td>Nah I don't think he goes to usf, he lives aro...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    data.shape




.. parsed-literal::

    (5572, 2)



.. code:: ipython3

    null_values = data.isnull().sum()
    
    
    null_percentage = (data.isnull().sum() / len(data)) * 100
    
    print("Null values in each column:")
    print(null_values)
    print("\nPercentage of null values in each column:")
    print(null_percentage)
    


.. parsed-literal::

    Null values in each column:
    v1    0
    v2    0
    dtype: int64
    
    Percentage of null values in each column:
    v1    0.0
    v2    0.0
    dtype: float64
    

.. code:: ipython3

    data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    data.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>label</th>
          <th>text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ham</td>
          <td>Go until jurong point, crazy.. Available only ...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ham</td>
          <td>Ok lar... Joking wif u oni...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>spam</td>
          <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ham</td>
          <td>U dun say so early hor... U c already then say...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ham</td>
          <td>Nah I don't think he goes to usf, he lives aro...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Using one-hot encoding
    one_hot_encoded_labels = pd.get_dummies(data['label'])
    print(one_hot_encoded_labels.head())
    
    #


.. parsed-literal::

         ham   spam
    0   True  False
    1   True  False
    2  False   True
    3   True  False
    4   True  False
    

.. code:: ipython3

    one_hot_encoded = pd.get_dummies(data['label'])
    one_hot_encoded = one_hot_encoded.astype(int)
    print(one_hot_encoded.head())
    print(data.head(15))
    


.. parsed-literal::

       0  1
    0  1  0
    1  1  0
    2  0  1
    3  1  0
    4  1  0
        label                                               text
    0       0  Go until jurong point, crazy.. Available only ...
    1       0                      Ok lar... Joking wif u oni...
    2       1  Free entry in 2 a wkly comp to win FA Cup fina...
    3       0  U dun say so early hor... U c already then say...
    4       0  Nah I don't think he goes to usf, he lives aro...
    5       1  FreeMsg Hey there darling it's been 3 week's n...
    6       0  Even my brother is not like to speak with me. ...
    7       0  As per your request 'Melle Melle (Oru Minnamin...
    8       1  WINNER!! As a valued network customer you have...
    9       1  Had your mobile 11 months or more? U R entitle...
    10      0  I'm gonna be home soon and i don't want to tal...
    11      1  SIX chances to win CASH! From 100 to 20,000 po...
    12      1  URGENT! You have won a 1 week FREE membership ...
    13      0  I've been searching for the right words to tha...
    14      0                I HAVE A DATE ON SUNDAY WITH WILL!!
    

.. code:: ipython3

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

.. code:: ipython3

    X = tfidf.fit_transform(data['text'])
    y = data['label']

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)


.. parsed-literal::

    (4457, 8404)
    (1115, 8404)
    

.. code:: ipython3

    model = MultinomialNB()
    model.fit(X_train, y_train)




.. raw:: html

    <style>#sk-container-id-1 {
      /* Definition of color scheme common for light and dark mode */
      --sklearn-color-text: black;
      --sklearn-color-line: gray;
      /* Definition of color scheme for unfitted estimators */
      --sklearn-color-unfitted-level-0: #fff5e6;
      --sklearn-color-unfitted-level-1: #f6e4d2;
      --sklearn-color-unfitted-level-2: #ffe0b3;
      --sklearn-color-unfitted-level-3: chocolate;
      /* Definition of color scheme for fitted estimators */
      --sklearn-color-fitted-level-0: #f0f8ff;
      --sklearn-color-fitted-level-1: #d4ebff;
      --sklearn-color-fitted-level-2: #b3dbfd;
      --sklearn-color-fitted-level-3: cornflowerblue;
    
      /* Specific color for light theme */
      --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
      --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-icon: #696969;
    
      @media (prefers-color-scheme: dark) {
        /* Redefinition of color scheme for dark theme */
        --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
        --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-icon: #878787;
      }
    }
    
    #sk-container-id-1 {
      color: var(--sklearn-color-text);
    }
    
    #sk-container-id-1 pre {
      padding: 0;
    }
    
    #sk-container-id-1 input.sk-hidden--visually {
      border: 0;
      clip: rect(1px 1px 1px 1px);
      clip: rect(1px, 1px, 1px, 1px);
      height: 1px;
      margin: -1px;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }
    
    #sk-container-id-1 div.sk-dashed-wrapped {
      border: 1px dashed var(--sklearn-color-line);
      margin: 0 0.4em 0.5em 0.4em;
      box-sizing: border-box;
      padding-bottom: 0.4em;
      background-color: var(--sklearn-color-background);
    }
    
    #sk-container-id-1 div.sk-container {
      /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
         but bootstrap.min.css set `[hidden] { display: none !important; }`
         so we also need the `!important` here to be able to override the
         default hidden behavior on the sphinx rendered scikit-learn.org.
         See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
      display: inline-block !important;
      position: relative;
    }
    
    #sk-container-id-1 div.sk-text-repr-fallback {
      display: none;
    }
    
    div.sk-parallel-item,
    div.sk-serial,
    div.sk-item {
      /* draw centered vertical line to link estimators */
      background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
      background-size: 2px 100%;
      background-repeat: no-repeat;
      background-position: center center;
    }
    
    /* Parallel-specific style estimator block */
    
    #sk-container-id-1 div.sk-parallel-item::after {
      content: "";
      width: 100%;
      border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
      flex-grow: 1;
    }
    
    #sk-container-id-1 div.sk-parallel {
      display: flex;
      align-items: stretch;
      justify-content: center;
      background-color: var(--sklearn-color-background);
      position: relative;
    }
    
    #sk-container-id-1 div.sk-parallel-item {
      display: flex;
      flex-direction: column;
    }
    
    #sk-container-id-1 div.sk-parallel-item:first-child::after {
      align-self: flex-end;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:last-child::after {
      align-self: flex-start;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:only-child::after {
      width: 0;
    }
    
    /* Serial-specific style estimator block */
    
    #sk-container-id-1 div.sk-serial {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--sklearn-color-background);
      padding-right: 1em;
      padding-left: 1em;
    }
    
    
    /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
    clickable and can be expanded/collapsed.
    - Pipeline and ColumnTransformer use this feature and define the default style
    - Estimators will overwrite some part of the style using the `sk-estimator` class
    */
    
    /* Pipeline and ColumnTransformer style (default) */
    
    #sk-container-id-1 div.sk-toggleable {
      /* Default theme specific background. It is overwritten whether we have a
      specific estimator or a Pipeline/ColumnTransformer */
      background-color: var(--sklearn-color-background);
    }
    
    /* Toggleable label */
    #sk-container-id-1 label.sk-toggleable__label {
      cursor: pointer;
      display: block;
      width: 100%;
      margin-bottom: 0;
      padding: 0.5em;
      box-sizing: border-box;
      text-align: center;
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:before {
      /* Arrow on the left of the label */
      content: "▸";
      float: left;
      margin-right: 0.25em;
      color: var(--sklearn-color-icon);
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
      color: var(--sklearn-color-text);
    }
    
    /* Toggleable content - dropdown */
    
    #sk-container-id-1 div.sk-toggleable__content {
      max-height: 0;
      max-width: 0;
      overflow: hidden;
      text-align: left;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content pre {
      margin: 0.2em;
      border-radius: 0.25em;
      color: var(--sklearn-color-text);
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted pre {
      /* unfitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
      /* Expand drop-down */
      max-height: 200px;
      max-width: 100%;
      overflow: auto;
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
      content: "▾";
    }
    
    /* Pipeline/ColumnTransformer-specific style */
    
    #sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator-specific style */
    
    /* Colorize estimator box */
    #sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label label.sk-toggleable__label,
    #sk-container-id-1 div.sk-label label {
      /* The background is the default theme color */
      color: var(--sklearn-color-text-on-default-background);
    }
    
    /* On hover, darken the color of the background */
    #sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    /* Label box, darken color on hover, fitted */
    #sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator label */
    
    #sk-container-id-1 div.sk-label label {
      font-family: monospace;
      font-weight: bold;
      display: inline-block;
      line-height: 1.2em;
    }
    
    #sk-container-id-1 div.sk-label-container {
      text-align: center;
    }
    
    /* Estimator-specific */
    #sk-container-id-1 div.sk-estimator {
      font-family: monospace;
      border: 1px dotted var(--sklearn-color-border-box);
      border-radius: 0.25em;
      box-sizing: border-box;
      margin-bottom: 0.5em;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    /* on hover */
    #sk-container-id-1 div.sk-estimator:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Specification for estimator info (e.g. "i" and "?") */
    
    /* Common style for "i" and "?" */
    
    .sk-estimator-doc-link,
    a:link.sk-estimator-doc-link,
    a:visited.sk-estimator-doc-link {
      float: right;
      font-size: smaller;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1em;
      height: 1em;
      width: 1em;
      text-decoration: none !important;
      margin-left: 1ex;
      /* unfitted */
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
      color: var(--sklearn-color-unfitted-level-1);
    }
    
    .sk-estimator-doc-link.fitted,
    a:link.sk-estimator-doc-link.fitted,
    a:visited.sk-estimator-doc-link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    div.sk-estimator:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover,
    div.sk-label-container:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover,
    div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    /* Span, style for the box shown on hovering the info icon */
    .sk-estimator-doc-link span {
      display: none;
      z-index: 9999;
      position: relative;
      font-weight: normal;
      right: .2ex;
      padding: .5ex;
      margin: .5ex;
      width: min-content;
      min-width: 20ex;
      max-width: 50ex;
      color: var(--sklearn-color-text);
      box-shadow: 2pt 2pt 4pt #999;
      /* unfitted */
      background: var(--sklearn-color-unfitted-level-0);
      border: .5pt solid var(--sklearn-color-unfitted-level-3);
    }
    
    .sk-estimator-doc-link.fitted span {
      /* fitted */
      background: var(--sklearn-color-fitted-level-0);
      border: var(--sklearn-color-fitted-level-3);
    }
    
    .sk-estimator-doc-link:hover span {
      display: block;
    }
    
    /* "?"-specific style due to the `<a>` HTML tag */
    
    #sk-container-id-1 a.estimator_doc_link {
      float: right;
      font-size: 1rem;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1rem;
      height: 1rem;
      width: 1rem;
      text-decoration: none;
      /* unfitted */
      color: var(--sklearn-color-unfitted-level-1);
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    #sk-container-id-1 a.estimator_doc_link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
    }
    </style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;MultinomialNB<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html">?<span>Documentation for MultinomialNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>MultinomialNB()</pre></div> </div></div></div></div>



.. code:: ipython3

    y_pred = model.predict(X_test)

.. code:: ipython3

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


.. parsed-literal::

    Accuracy: 0.968609865470852
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.96      1.00      0.98       965
               1       1.00      0.77      0.87       150
    
        accuracy                           0.97      1115
       macro avg       0.98      0.88      0.93      1115
    weighted avg       0.97      0.97      0.97      1115
    
    

.. code:: ipython3

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

.. code:: ipython3

    cm = confusion_matrix(y_test, y_pred)
    print(cm)


.. parsed-literal::

    [[965   0]
     [ 35 115]]
    

.. code:: ipython3

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()



.. image:: output_16_0.png


.. code:: ipython3

    """# Plotting actual vs predicted values
    plt.figure(figsize=(10, 5))
    
    # Create a DataFrame to hold actual and predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    
    # Plot actual values
    plt.plot(results_df.index, results_df['actual'], marker='o', label='Actual', color='blue')
    
    # Plot predicted values
    plt.plot(results_df.index, results_df['predicted'], marker='x', label='Predicted', color='orange')
    
    # Adding labels and title
    plt.title('Actual vs Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Label (0 = Ham, 1 = Spam)')
    plt.xticks(results_df.index)  
    plt.yticks([0, 1])     
    plt.legend()
    plt.grid()
    plt.show()"""




.. parsed-literal::

    "# Plotting actual vs predicted values\nplt.figure(figsize=(10, 5))\n\n# Create a DataFrame to hold actual and predicted values\nresults_df = pd.DataFrame({\n    'actual': y_test,\n    'predicted': y_pred\n})\n\n# Plot actual values\nplt.plot(results_df.index, results_df['actual'], marker='o', label='Actual', color='blue')\n\n# Plot predicted values\nplt.plot(results_df.index, results_df['predicted'], marker='x', label='Predicted', color='orange')\n\n# Adding labels and title\nplt.title('Actual vs Predicted Labels')\nplt.xlabel('Sample Index')\nplt.ylabel('Label (0 = Ham, 1 = Spam)')\nplt.xticks(results_df.index)  \nplt.yticks([0, 1])     \nplt.legend()\nplt.grid()\nplt.show()"



.. code:: ipython3

    new_messages = ["How are you", "Hey, are we still on for lunch today?", "you had win a mega offer,click  here to win", "Can you send me the report by tonight?","you had recieve the price please contact us to recieve your price"]
    
    new_messages_transformed = tfidf.transform(new_messages)
    predictions = model.predict(new_messages_transformed)
    for msg, label in zip(new_messages, predictions):
        print(f"Message: {msg} -> Prediction: {'Spam' if label == 1 else 'Ham'}")
    


.. parsed-literal::

    Message: How are you -> Prediction: Ham
    Message: Hey, are we still on for lunch today? -> Prediction: Ham
    Message: you had win a mega offer,click  here to win -> Prediction: Spam
    Message: Can you send me the report by tonight? -> Prediction: Ham
    Message: you had recieve the price please contact us to recieve your price -> Prediction: Ham
    

.. code:: ipython3

    import joblib
    
    
    joblib.dump(model, r'C:\Users\agri2024\Downloads\PP6\spam_detection_model.pkl')  # Save the model as a .pkl file
    print("Saving Successful")


.. parsed-literal::

    Saving Successful
    

.. code:: ipython3

    # Load the model from the file
    with open('spam_detection_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    # Now you can use loaded_model to make predictions
    y_pred_loaded = loaded_model.predict(X_test_vectorized)
    
