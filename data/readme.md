### 1.Raw Data Sources

1.1 JPCERT/CC - Japan Computer Emergency Response Team Coordination Center
It provides security alerts, phishing-related data, and other cybersecurity intelligence. In this repository, we include the latest phishing URLs identified by JPCERT.

File Names: 202409, 202410

Description: These files contain a list of phishing URLs reported in specific incidents as per JPCERT's latest security findings.
 
1.2 Legit_datasets (University of New Brunswick)
This dataset includes a collection of URLs marked as legitimate, curated by the University of New Brunswick for use in phishing detection research - https://www.unb.ca/cic/datasets/url-2016.html

File Names: Legit_datasets.csv

Description: A comprehensive list of legitimate websites used for training, evaluation, or comparison purposes in phishing detection models.

1.3 OpenPhish Feed
OpenPhish provides real-time phishing feed data, offering a regularly updated list of known phishing websites.

File Name: feed.txt

Description: This file contains the latest phishing URLs collected by OpenPhish, ideal for identifying active phishing campaigns - https://www.openphish.com/phishing_feeds.html

### 2.Processed Data

2.1 Phishing Data : 5000 random data 

2.2 Legitimate Data : 5000 random data 

2.3 GAN-Phishing data & GAN-Legitimate data : 2000 from each category

### Features Extraction explanation

**Address Bar-Based Features**

1. **Domain of the URL**:
   - **Description**: The domain name reveals the website’s origin. Phishing sites often use misleading or similar-looking domains to impersonate legitimate websites.

2. **IP Address in the URL**:
   - **Description**: URLs containing an IP address instead of a domain name are suspicious. Legitimate websites generally use domain names, making IP-based URLs a potential phishing red flag.

3. **Presence of "@" Symbol in the URL**:
   - **Description**: The "@" symbol in URLs can hide part of the domain or mislead users, often seen in phishing URLs to disguise the true destination.

4. **Length of the URL**:
   - **Description**: Longer URLs are often used in phishing attacks to obscure the destination or include misleading parameters to deceive users.

5. **Depth of the URL**:
   - **Description**: The number of subdirectories or folders in a URL. Phishing sites may use deeper URL structures to confuse users and mimic legitimate websites.

6. **Redirection "//" in the URL**:
   - **Description**: Multiple consecutive slashes (e.g., `//`) in the URL path can indicate redirection or an attempt to obscure the true destination, often found in phishing websites.

7. **HTTP/HTTPS in the Domain Name**:
   - **Description**: Phishing URLs may insert `http` or `https` within the domain name (e.g., `examplehttp.com`) to mislead users into thinking the site is secure.

8. **Use of URL Shortening Services (e.g., TinyURL)**:
   - **Description**: Shortened URLs hide the true destination, making it harder for users to identify malicious links, commonly used in phishing campaigns.

9. **Prefix or Suffix "-" in the Domain**:
   - **Description**: A hyphen in the domain name is often used in phishing URLs to mimic trusted sites, adding confusion by making small alterations to popular domains.

**Domain-Based Features**

1. **DNS Record**:
   - **Description**: A valid DNS record is necessary for a domain to be reachable. Phishing sites may lack valid DNS records or use temporary domains that are hard to trace.

2. **Age of Domain**:
   - **Description**: Legitimate websites typically have older domains, whereas phishing sites often use newly registered domains to evade detection.

3. **End Period of Domain**:
   - **Description**: The expiration date of a domain can indicate its legitimacy. Phishing websites may use domains with short or soon-to-expire registration periods to avoid being detected over time. 

**HTML and Javascript based features**

1. **IFrame Redirection**:
   - **Description**: Phishing websites may use hidden or invisible IFrames to load malicious content, redirect users to fake pages, or capture credentials without the user’s knowledge.

2. **Status Bar Customization**:
   - **Description**: Using JavaScript, phishing websites can alter the browser’s status bar to disguise links, making them appear safe to users when they are actually malicious.

3. **Disabling Right Click**:
   - **Description**: Some phishing sites disable right-click functionality to prevent users from inspecting the page, viewing source code, or using browser tools to reveal malicious behavior.

4. **Website Forwarding**:
   - **Description**: Phishing websites often automatically redirect users to a fake login page or malicious site using JavaScript or HTML meta-refresh tags to steal sensitive information.
