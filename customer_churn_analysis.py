# Customer Churn Analysis - Labmentix Project
# Author: [Payal Madam]
# Date: July 23, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerChurnAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None
        self.label_encoders = {}
        
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic customer data for analysis"""
        np.random.seed(42)
        
        # Generate customer features
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure_months': np.random.exponential(24, n_samples).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': lambda x: x['tenure_months'] * x['monthly_charges'] + np.random.normal(0, 100, n_samples),
            'contract_length': np.random.choice(['Month-to-month', '1-year', '2-year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples, p=[0.4, 0.5, 0.1]),
            'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'streaming_tv': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
            'streaming_movies': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        df['total_charges'] = df['tenure_months'] * df['monthly_charges'] + np.random.normal(0, 100, n_samples)
        
        # Create churn based on realistic factors
        churn_probability = (
            0.1 +  # Base churn rate
            (df['contract_length'] == 'Month-to-month') * 0.3 +  # Month-to-month more likely to churn
            (df['tenure_months'] < 12) * 0.2 +  # New customers more likely to churn
            (df['monthly_charges'] > 80) * 0.15 +  # High charges increase churn
            (df['tech_support'] == 'No') * 0.1 +  # No tech support increases churn
            (df['senior_citizen'] == 1) * 0.05 +  # Senior citizens slightly more likely
            np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        # Ensure probabilities are between 0 and 1
        churn_probability = np.clip(churn_probability, 0, 1)
        df['churn'] = np.random.binomial(1, churn_probability, n_samples)
        
        self.data = df
        return df
    
    def load_data(self, filepath=None):
        """Load data from file or generate sample data"""
        if filepath:
            self.data = pd.read_csv(filepath)
        else:
            self.data = self.generate_sample_data()
        
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("=" * 60)
        print("CUSTOMER CHURN ANALYSIS - EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("\n1. DATASET OVERVIEW:")
        print(f"   • Total Customers: {len(self.data):,}")
        print(f"   • Features: {self.data.shape[1]}")
        print(f"   • Churned Customers: {self.data['churn'].sum():,} ({self.data['churn'].mean()*100:.1f}%)")
        print(f"   • Retained Customers: {(len(self.data) - self.data['churn'].sum()):,} ({(1-self.data['churn'].mean())*100:.1f}%)")
        
        # Missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n2. MISSING VALUES: {missing_values.sum()} total")
            print(missing_values[missing_values > 0])
        else:
            print("\n2. MISSING VALUES: None found ✓")
        
        # Churn analysis by key features
        print("\n3. CHURN ANALYSIS BY KEY FEATURES:")
        
        categorical_features = ['gender', 'contract_length', 'payment_method', 'internet_service']
        for feature in categorical_features:
            churn_rate = self.data.groupby(feature)['churn'].mean().sort_values(ascending=False)
            print(f"\n   {feature.upper()}:")
            for category, rate in churn_rate.items():
                print(f"   • {category}: {rate*100:.1f}% churn rate")
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = self.data['churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0,0].pie(churn_counts.values, labels=['Retained', 'Churned'], colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Overall Churn Distribution', fontweight='bold')
        
        # 2. Churn by Contract Length
        contract_churn = pd.crosstab(self.data['contract_length'], self.data['churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', ax=axes[0,1], color=colors)
        axes[0,1].set_title('Churn Rate by Contract Length', fontweight='bold')
        axes[0,1].set_ylabel('Churn Rate (%)')
        axes[0,1].legend(['Retained', 'Churned'])
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Monthly Charges Distribution
        churned = self.data[self.data['churn'] == 1]['monthly_charges']
        retained = self.data[self.data['churn'] == 0]['monthly_charges']
        axes[1,0].hist([retained, churned], bins=30, alpha=0.7, label=['Retained', 'Churned'], color=colors)
        axes[1,0].set_title('Monthly Charges Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Monthly Charges ($)')
        axes[1,0].set_ylabel('Number of Customers')
        axes[1,0].legend()
        
        # 4. Tenure vs Churn
        tenure_churn = self.data.groupby(pd.cut(self.data['tenure_months'], bins=10))['churn'].mean() * 100
        tenure_churn.plot(kind='bar', ax=axes[1,1], color='#3498db')
        axes[1,1].set_title('Churn Rate by Tenure', fontweight='bold')
        axes[1,1].set_ylabel('Churn Rate (%)')
        axes[1,1].set_xlabel('Tenure (Months)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 5. Correlation Heatmap
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[2,0])
        axes[2,0].set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 6. Churn by Internet Service
        internet_churn = pd.crosstab(self.data['internet_service'], self.data['churn'], normalize='index') * 100
        internet_churn.plot(kind='bar', ax=axes[2,1], color=colors)
        axes[2,1].set_title('Churn Rate by Internet Service', fontweight='bold')
        axes[2,1].set_ylabel('Churn Rate (%)')
        axes[2,1].legend(['Retained', 'Churned'])
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('customer_churn_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved as 'customer_churn_analysis.png'")
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        # Create a copy for preprocessing
        df_ml = self.data.copy()
        
        # Encode categorical variables
        categorical_columns = df_ml.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'customer_id']
        
        for column in categorical_columns:
            le = LabelEncoder()
            df_ml[column] = le.fit_transform(df_ml[column])
            self.label_encoders[column] = le
        
        # Select features (exclude customer_id and target variable)
        feature_columns = [col for col in df_ml.columns if col not in ['customer_id', 'churn']]
        X = df_ml[feature_columns]
        y = df_ml['churn']
        
        return X, y
    
    def train_model(self):
        """Train machine learning model for churn prediction"""
        print("\n" + "=" * 60)
        print("MACHINE LEARNING MODEL TRAINING")
        print("=" * 60)
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   • Accuracy: {accuracy*100:.2f}%")
        
        print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(top_features['feature'], top_features['importance'], color='#3498db')
        plt.title('Top 10 Feature Importance for Churn Prediction', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, feature_importance
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n" + "=" * 60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        # Calculate key metrics
        total_customers = len(self.data)
        churned_customers = self.data['churn'].sum()
        churn_rate = churned_customers / total_customers * 100
        avg_monthly_revenue = self.data['monthly_charges'].mean()
        avg_customer_lifetime = self.data['tenure_months'].mean()
        
        # Revenue impact
        monthly_revenue_loss = churned_customers * avg_monthly_revenue
        annual_revenue_loss = monthly_revenue_loss * 12
        
        print(f"\n💰 FINANCIAL IMPACT:")
        print(f"   • Monthly Revenue Loss: ${monthly_revenue_loss:,.2f}")
        print(f"   • Annual Revenue Loss: ${annual_revenue_loss:,.2f}")
        print(f"   • Average Customer Lifetime: {avg_customer_lifetime:.1f} months")
        
        # High-risk segments
        high_risk = self.data[
            (self.data['contract_length'] == 'Month-to-month') & 
            (self.data['tenure_months'] < 12)
        ]
        
        print(f"\n⚠️  HIGH-RISK CUSTOMER SEGMENT:")
        print(f"   • Month-to-month customers with <12 months tenure")
        print(f"   • Total customers in this segment: {len(high_risk):,}")
        print(f"   • Churn rate in this segment: {high_risk['churn'].mean()*100:.1f}%")
        
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        print(f"   1. TARGET RETENTION CAMPAIGNS:")
        print(f"      • Focus on month-to-month contract customers")
        print(f"      • Implement early intervention for customers <6 months tenure")
        print(f"      • Offer incentives for longer contract commitments")
        
        print(f"\n   2. SERVICE IMPROVEMENTS:")
        print(f"      • Enhance tech support services (high correlation with retention)")
        print(f"      • Review pricing strategy for high monthly charge customers")
        print(f"      • Improve fiber optic service quality")
        
        print(f"\n   3. PROACTIVE MEASURES:")
        print(f"      • Implement churn prediction model in production")
        print(f"      • Create automated alerts for high-risk customers")
        print(f"      • Develop personalized retention offers")
        
        print(f"\n   4. EXPECTED IMPACT:")
        print(f"      • 10% reduction in churn could save ${annual_revenue_loss*0.1:,.2f} annually")
        print(f"      • Focus on top 20% at-risk customers for maximum ROI")

def main():
    """Main execution function"""
    print("🚀 Starting Customer Churn Analysis Project")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CustomerChurnAnalyzer()
    
    # Load and explore data
    data = analyzer.load_data()
    analyzer.explore_data()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Train machine learning model
    accuracy, feature_importance = analyzer.train_model()
    
    # Generate business insights
    analyzer.generate_business_insights()
    
    print(f"\n✅ Analysis Complete!")
    print(f"   📊 Visualizations saved as PNG files")
    print(f"   🤖 Model accuracy: {accuracy*100:.2f}%")
    print(f"   💡 Business recommendations generated")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()