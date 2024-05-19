{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9c7eaa95",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e8d7c03a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15624510</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15810944</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15668575</td>\n",
       "      <td>Female</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15603246</td>\n",
       "      <td>Female</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15804002</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>15691863</td>\n",
       "      <td>Female</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>15706071</td>\n",
       "      <td>Male</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>15654296</td>\n",
       "      <td>Female</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>15755018</td>\n",
       "      <td>Male</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>15594041</td>\n",
       "      <td>Female</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      User ID  Gender  Age  EstimatedSalary  Purchased\n",
       "0    15624510    Male   19            19000          0\n",
       "1    15810944    Male   35            20000          0\n",
       "2    15668575  Female   26            43000          0\n",
       "3    15603246  Female   27            57000          0\n",
       "4    15804002    Male   19            76000          0\n",
       "..        ...     ...  ...              ...        ...\n",
       "395  15691863  Female   46            41000          1\n",
       "396  15706071    Male   51            23000          1\n",
       "397  15654296  Female   50            20000          1\n",
       "398  15755018    Male   36            33000          0\n",
       "399  15594041  Female   49            36000          1\n",
       "\n",
       "[400 rows x 5 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"Social_Network_Ads.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3ee555b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Temp\\ipykernel_3268\\3596937014.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['Gender'][df['Gender'] == 'Male'] = 1\n",
      "C:\\Users\\HP\\AppData\\Local\\Temp\\ipykernel_3268\\3596937014.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['Gender'][df['Gender'] == 'Female'] = 0\n"
     ]
    }
   ],
   "source": [
    "df['Gender'][df['Gender'] == 'Male'] = 1\n",
    "df['Gender'][df['Gender'] == 'Female'] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "48f76704",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15624510</td>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15810944</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15668575</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15603246</td>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15804002</td>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>15691863</td>\n",
       "      <td>0</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>15706071</td>\n",
       "      <td>1</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>15654296</td>\n",
       "      <td>0</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>15755018</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>15594041</td>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      User ID Gender  Age  EstimatedSalary  Purchased\n",
       "0    15624510      1   19            19000          0\n",
       "1    15810944      1   35            20000          0\n",
       "2    15668575      0   26            43000          0\n",
       "3    15603246      0   27            57000          0\n",
       "4    15804002      1   19            76000          0\n",
       "..        ...    ...  ...              ...        ...\n",
       "395  15691863      0   46            41000          1\n",
       "396  15706071      1   51            23000          1\n",
       "397  15654296      0   50            20000          1\n",
       "398  15755018      1   36            33000          0\n",
       "399  15594041      0   49            36000          1\n",
       "\n",
       "[400 rows x 5 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a5beac4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    257\n",
       "1    143\n",
       "Name: Purchased, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Purchased'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "9767563b",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop('Purchased', axis=1)\n",
    "y = df['Purchased']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "e27cabe7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(280, 4) (120, 4) (280,) (120,)\n"
     ]
    }
   ],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)\n",
    "print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3a36489d",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier = SVC(kernel = 'linear')\n",
    "classifier.fit(x_train, y_train)\n",
    "y_pred = classifier.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b520ea2a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,\n",
       "       0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e705ad8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score: 0.7166666666666667\n",
      "Confusion Matrix:\n",
      "[[70  3]\n",
      " [31 16]]\n"
     ]
    }
   ],
   "source": [
    "print(f\"Accuracy Score: {accuracy_score(y_test, y_pred)}\")\n",
    "print(f\"Confusion Matrix:\\n{confusion_matrix(y_test, y_pred)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f9f1f50e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.scatter(x_test['EstimatedSalary'], x_test['Age'], c=y_pred)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
