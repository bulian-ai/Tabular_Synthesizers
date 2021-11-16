**10/10/2021**
- 
>- Rewrite CTGAN classes as per Videha nomenclature (needs detailed class names/func names) - **Deadline: 31/10/2021**
>- Integrate Fast.ai's LR finder - **Deadline: 17/10/2021** 
>- Integrate Adversarial training to filter synthetic rows similar to original dataset - **Deadline: 25/10/2021**
>- Multi GPU support - Medium Priority **Deadline: 31/10/2021**
>- Package as installable library (docker stuff?) **Deadline 15/11/2021**
>- Test on Linux/MAC/Windows (use Jarvislabs.ai for some of these, and local windows/mac) **Deadline 22/11/2021**
>- Establish new benchmarks on real world Kaggle datasets for Pharma/Finance **Deadline 30/11/2021**
>>- See if there are any novel fintech datasets such as numer.ai
>- Establish SOTA using noisy training mechanisms using Real+Synthetic datasets (https://arxiv.org/abs/2109.14563) **Go-No-ho Decision: 1/12/2021**
>- Compile new commercial license

**12/10/2021: Meeting notes**
-
>- Fit method should return a dict of model, train data, test data; in case adv_samples=False, then train_data and test_data should have none values
>- model.sample method should take in train_data, test_data if adv_samples=True, else, None  

**14/10/2021: Meeting notes**
-
>- Metrics and viz

**16/10/2021: Meeting notes**

>- ~~Integrate Fast.ai's LR finder - **Deadline: 17/10/2021**~~
>- Integrate Adversarial training to filter synthetic rows similar to original dataset - **AY: Deadline: 20/10/2021**
>- In adversarial training change fit to fit_adversarial and sample to sample_advarsarial
>- Model objects from adv train should contain modified dict with train and test indexes; sample_adv should be able to take in these new items directly from new model dict
>- Rewrite CTGAN classes as per Videha nomenclature (needs detailed class names/func names) - **RT: Deadline: 31/10/2021**
![image](https://user-images.githubusercontent.com/16912628/137634567-f5cdabe1-080b-4e93-b0af-8f82d9cda7b8.png)

**24/10/2021: Meeting notes**
>- Explore private library installation methods eg: authoentication, token generation etc
>- **Results and vizualization** methods: **Deadline 8/11/2021**
>>- CTGAN inbuilt metrics
>>- Metrics for Adversarial sampling
>>- More recent metrics for synthetic datasets
>- Noisy label training code or just documentation to be decided on 1st Dec
>- Full Documentation of functionality and code:
>>- PDF/Website based client login/Custom made jupyter notebooks? **Decision deadline 1/12/2021**
