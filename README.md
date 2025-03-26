# Engineering and Evaluation of AI. Continuous Assessment

# CA_multi-label_email_classification
 implement Chained Multi-output classification.

## How to run
1. Install dependencies: `pip install pandas numpy stanza transformers scikit-learn`
2. Run evaluation: `python main.py`, or alternatively, open notebook `test.ipybn` and explore the results of a test run.

# Results 
```
Evaluation for AppGallery &amp; Games , random forest

                                                                                        precision    recall  f1-score   support
------------------------------------------------------------------------------------------------------------------------------
('Others', nan, nan)                                                                         0.67      0.75      0.71         8
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't update Apps")                         0.00      0.00      0.00         1
('Suggestion', 'AppGallery-Use', 'Others')                                                   0.00      0.00      0.00         0
('Problem/Fault', 'Third Party APPs', 'Refund')                                              1.00      1.00      1.00         1
('Suggestion', 'VIP / Offers / Promotions', 'Offers / Vouchers / Promotions')                1.00      0.50      0.67         2
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't install Apps")                        1.00      1.00      1.00         4
('Problem/Fault', 'AppGallery-Install/Upgrade', 'Other download/install/update issue')       1.00      1.00      1.00         1
('Suggestion', 'General', 'Personal data')                                                   0.33      1.00      0.50         1
('Problem/Fault', 'Coupon/Gifts/Points Issues', "Can't use or acquire")                      0.67      1.00      0.80         2
('Problem/Fault', 'AppGallery-Use', 'UI Abnormal in Huawei AppGallery')                      1.00      0.50      0.67         2
('Suggestion', 'General', 'Others')                                                          0.00      0.00      0.00         1
('Problem/Fault', 'General', 'Security issue / malware')                                     0.00      0.00      0.00         1
------------------------------------------------------------------------------------------------------------------------------
accuracy                                                                                     0.71        72
------------------------------------------------------------------------------------------------------------------------------
macro avg                                                                                    0.56      0.56      0.53        24
weighted avg                                                                                 0.71      0.71      0.68        24
Evaluation for AppGallery &amp; Games , logistic regression

                                                                                        precision    recall  f1-score   support
------------------------------------------------------------------------------------------------------------------------------
('Others', nan, nan)                                                                         0.71      0.62      0.67         8
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't update Apps")                         0.00      0.00      0.00         1
('Problem/Fault', 'Third Party APPs', 'Refund')                                              1.00      1.00      1.00         1
('Suggestion', 'VIP / Offers / Promotions', 'Offers / Vouchers / Promotions')                1.00      1.00      1.00         2
('Problem/Fault', 'General', 'Cannot connect - Server')                                      0.00      0.00      0.00         0
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't install Apps")                        1.00      0.75      0.86         4
('Problem/Fault', 'AppGallery-Install/Upgrade', 'Other download/install/update issue')       1.00      1.00      1.00         1
('Suggestion', 'General', 'Personal data')                                                   0.50      1.00      0.67         1
('Problem/Fault', 'Coupon/Gifts/Points Issues', "Can't use or acquire")                      1.00      1.00      1.00         2
('Problem/Fault', 'AppGallery-Use', 'UI Abnormal in Huawei AppGallery')                      1.00      1.00      1.00         2
('Suggestion', 'General', 'Others')                                                          0.50      1.00      0.67         1
('Problem/Fault', 'General', 'Security issue / malware')                                     0.00      0.00      0.00         1
('Problem/Fault', 'Coupon/Gifts/Points Issues', 'Cooperated campaign issue')                 0.00      0.00      0.00         0
------------------------------------------------------------------------------------------------------------------------------
accuracy                                                                                     0.75        72
------------------------------------------------------------------------------------------------------------------------------
macro avg                                                                                    0.59      0.64      0.60        24
weighted avg                                                                                 0.78      0.75      0.75        24
Evaluation for AppGallery &amp; Games , dnn

                                                                                        precision    recall  f1-score   support
------------------------------------------------------------------------------------------------------------------------------
('Others', nan, nan)                                                                         0.78      0.88      0.82         8
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't update Apps")                         0.00      0.00      0.00         1
('Suggestion', 'AppGallery-Use', 'Others')                                                   0.00      0.00      0.00         0
('Problem/Fault', 'Third Party APPs', 'Refund')                                              1.00      1.00      1.00         1
('Suggestion', 'VIP / Offers / Promotions', 'Offers / Vouchers / Promotions')                1.00      1.00      1.00         2
('Problem/Fault', 'AppGallery-Install/Upgrade', "Can't install Apps")                        1.00      1.00      1.00         4
('Problem/Fault', 'AppGallery-Install/Upgrade', 'Other download/install/update issue')       0.50      1.00      0.67         1
('Suggestion', 'General', 'Personal data')                                                   1.00      1.00      1.00         1
('Problem/Fault', 'Coupon/Gifts/Points Issues', "Can't use or acquire")                      1.00      0.50      0.67         2
('Problem/Fault', 'AppGallery-Use', 'UI Abnormal in Huawei AppGallery')                      1.00      0.50      0.67         2
('Suggestion', 'General', 'Others')                                                          0.00      0.00      0.00         1
('Problem/Fault', 'General', 'Security issue / malware')                                     0.00      0.00      0.00         1
('Problem/Fault', 'Coupon/Gifts/Points Issues', 'Cooperated campaign issue')                 0.00      0.00      0.00         0
------------------------------------------------------------------------------------------------------------------------------
accuracy                                                                                     0.75        72
------------------------------------------------------------------------------------------------------------------------------
macro avg                                                                                    0.56      0.53      0.52        24
weighted avg                                                                                 0.78      0.75      0.75        24
Evaluation for In-App Purchase , random forest

                                                        precision    recall  f1-score   support
----------------------------------------------------------------------------------------------
('Suggestion', 'Payment', 'Subscription cancellation')       0.60      1.00      0.75         9
('Suggestion', 'Other', nan)                                 0.00      0.00      0.00         3
('Suggestion', 'Payment', 'Query deduction details')         0.00      0.00      0.00         2
('Suggestion', 'Invoice', 'Invoice related request')         1.00      1.00      1.00         1
('Problem/Fault', 'Payment issue', 'Risk Control')           1.00      0.50      0.67         2
----------------------------------------------------------------------------------------------
accuracy                                                     0.65        51
----------------------------------------------------------------------------------------------
macro avg                                                    0.52      0.50      0.48        17
weighted avg                                                 0.49      0.65      0.53        17
Evaluation for In-App Purchase , logistic regression

                                                        precision    recall  f1-score   support
----------------------------------------------------------------------------------------------
('Suggestion', 'Payment', 'Subscription cancellation')       0.75      1.00      0.86         9
('Suggestion', 'Other', nan)                                 1.00      1.00      1.00         3
('Suggestion', 'Payment', 'Query deduction details')         0.00      0.00      0.00         2
('Suggestion', 'Invoice', 'Invoice related request')         1.00      1.00      1.00         1
('Problem/Fault', 'Payment issue', 'Risk Control')           1.00      0.50      0.67         2
----------------------------------------------------------------------------------------------
accuracy                                                     0.82        51
----------------------------------------------------------------------------------------------
macro avg                                                    0.75      0.70      0.70        17
weighted avg                                                 0.75      0.82      0.77        17
Evaluation for In-App Purchase , dnn

                                                        precision    recall  f1-score   support
----------------------------------------------------------------------------------------------
('Suggestion', 'Payment', 'Subscription cancellation')       0.75      1.00      0.86         9
('Suggestion', 'Other', nan)                                 0.00      0.00      0.00         3
('Suggestion', 'Payment', 'Query deduction details')         0.00      0.00      0.00         2
('Suggestion', 'Invoice', 'Invoice related request')         1.00      1.00      1.00         1
('Problem/Fault', 'Payment issue', 'Risk Control')           1.00      0.50      0.67         2
----------------------------------------------------------------------------------------------
accuracy                                                     0.65        51
----------------------------------------------------------------------------------------------
macro avg                                                    0.55      0.50      0.50        17
weighted avg                                                 0.57      0.65      0.59        17

```