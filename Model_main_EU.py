import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize


source_data = "Statistical Review of World Energy Data.xlsx"  
source_data_ethanol = "HATCH_1.0_modified.xlsx"  


df_solar = pd.read_excel(source_data, sheet_name="Solar Installed Capacity", header=None)
df_wind = pd.read_excel(source_data, sheet_name="Wind Installed Capacity", header=None)
df_biofuels = pd.read_excel(source_data, sheet_name="Biofuels production - kboed", header=None)
df_ethanol = pd.read_excel(source_data_ethanol, sheet_name="HATCH_1.0", header=None)

solar_years = df_solar.iloc[3, 1:25].values  
wind_years = df_wind.iloc[3, 1:28].values  
biofuels_years = df_biofuels.iloc[2, 5:35].values  
ethanol_years = df_ethanol.iloc[0, 280:306].values  

solar_capacity = df_solar.iloc[42, 1:25].values
wind_capacity = df_wind.iloc[39, 1:28].values
biofuels_capacity = df_biofuels.iloc[28, 5:35].values
ethanol_capacity = df_ethanol.iloc[74, 280:306].values  


# Function to fit the exponential growth model
def exp_growth_solar(x, a, b):
    return a * (1 + b) ** (x-solar_years[0]-1)

def exp_growth_wind(x, a, b):
    return a * (1 + b) ** (x-wind_years[0]-1)

def exp_growth_biofuels(x, a, b):
    return a * (1 + b) ** (x-biofuels_years[0]-1)

def exp_growth_ethanol(x, a, b):
    return a * (1 + b) ** (x-ethanol_years[0]-1)

time_slice_length = 5

# List to store results
results_solar = []
results_wind = []
results_biofuels = []
results_ethanol = []
results_oil = []

for start in range(len(solar_years) - time_slice_length + 1):
    slice_years = solar_years[start:start + time_slice_length]
    slice_capacity = solar_capacity[start:start + time_slice_length]
    
    popt, _ = curve_fit(exp_growth_solar, slice_years, slice_capacity, p0=[100, 0.4])
    
    a, b = popt
    
    predicted_capacity = exp_growth_solar(slice_years, *popt)
    
    results_solar.append({
        'slice_length': time_slice_length,
        'slice_start': slice_years[0],
        'a': a,
        'b': b,  
        'growth_rate': b,  
        'predicted_capacity': predicted_capacity
    })

for start in range(len(wind_years) - time_slice_length + 1):
    slice_years = wind_years[start:start + time_slice_length]
    slice_capacity = wind_capacity[start:start + time_slice_length]
    
    popt, _ = curve_fit(exp_growth_wind, slice_years, slice_capacity, p0=[100, 0.4])
    
    a, b = popt
    
    predicted_capacity = exp_growth_wind(slice_years, *popt)
    
    results_wind.append({
        'slice_length': time_slice_length,
        'slice_start': slice_years[0],
        'a': a,
        'b': b,  
        'growth_rate': b,  
        'predicted_capacity': predicted_capacity
    })

for start in range(len(biofuels_years) - time_slice_length + 1):
    slice_years = biofuels_years[start:start + time_slice_length]
    slice_capacity = biofuels_capacity[start:start + time_slice_length]
    
    popt, _ = curve_fit(exp_growth_biofuels, slice_years, slice_capacity, p0=[100, 0.4])
    
    a, b = popt
    
    predicted_capacity = exp_growth_biofuels(slice_years, *popt)
    
    results_biofuels.append({
        'slice_length': time_slice_length,
        'slice_start': slice_years[0],
        'a': a,
        'b': b,  
        'growth_rate': b,  
        'predicted_capacity': predicted_capacity
    })

for start in range(len(ethanol_years) - time_slice_length + 1):
    slice_years = ethanol_years[start:start + time_slice_length]
    slice_capacity = ethanol_capacity[start:start + time_slice_length]
    
    popt, _ = curve_fit(exp_growth_ethanol, slice_years, slice_capacity, p0=[100, 0.4])
    
    a, b = popt
    
    predicted_capacity = exp_growth_ethanol(slice_years, *popt)
    
    results_ethanol.append({
        'slice_length': time_slice_length,
        'slice_start': slice_years[0],
        'a': a,
        'b': b,  
        'growth_rate': b,  
        'predicted_capacity': predicted_capacity
    })


df_results_solar = pd.DataFrame(results_solar)
df_results_wind = pd.DataFrame(results_wind)
df_results_biofuels = pd.DataFrame(results_biofuels)
df_results_ethanol = pd.DataFrame(results_ethanol)

solar_growth_by_year = df_results_solar[['slice_start', 'growth_rate']]
wind_growth_by_year = df_results_wind[['slice_start', 'growth_rate']]
biofuels_growth_by_year = df_results_biofuels[['slice_start', 'growth_rate']]
ethanol_growth_by_year = df_results_ethanol[['slice_start', 'growth_rate']]

selected_solar_years = [2000, 2001,2002,2003,2004,2005,2006,2007,2008,2009]  
solar_selected = df_results_solar[df_results_solar['slice_start'].isin(selected_solar_years)]
solar_growth_rates = np.array(solar_selected['growth_rate'].values)


selected_wind_years = [1997, 1998,1999,2000,2001,2002,2003,2004,2005,2006]  
wind_selected = df_results_wind[df_results_wind['slice_start'].isin(selected_wind_years)]
wind_growth_rates = np.array(wind_selected['growth_rate'].values)

selected_biofuel_years = [1996,1997,1998,1999,2000,2001,2002,2003,2004,2005]  
biofuels_selected = df_results_biofuels[df_results_biofuels['slice_start'].isin(selected_biofuel_years)]
biofuels_growth_rates = np.array(biofuels_selected['growth_rate'].values)

selected_ethanol_years = [1999,2000,2001,2002,2003,2004,2005,2006,2007]  
ethanol_selected = df_results_ethanol[df_results_ethanol['slice_start'].isin(selected_ethanol_years)]
ethanol_growth_rates = np.array(ethanol_selected['growth_rate'].values)



# Combine growth rates
# FOR BIOFUELS/ETHANOL: substitute solar and wind with biofuels and ethanol and run the code 
combined_growth_rates = np.concatenate([solar_growth_rates, wind_growth_rates])

# Define parameters for the truncated normal distribution for growth rates
a_growth = np.min(combined_growth_rates)  
b_growth = np.max(combined_growth_rates)  
mean_growth = np.mean(combined_growth_rates)  
std_growth = np.std(combined_growth_rates)  

a_std_growth = (a_growth - mean_growth) / std_growth
b_std_growth = (b_growth - mean_growth) / std_growth

def truncated_normal(a, b, cap_1, cumprob_likely, likely2025_mean):
    def fn(x):
        root1 = (
            x[0] + x[1] * 
            (-norm.pdf((b - x[0]) / x[1]) + norm.pdf((a - x[0]) / x[1])) / 
            (norm.cdf((b - x[0]) / x[1]) - norm.cdf((a - x[0]) / x[1])) - likely2025_mean
        )
        root2 = (
            (norm.cdf((cap_1 - x[0]) / x[1]) - norm.cdf((a - x[0]) / x[1])) / 
            (norm.cdf((b - x[0]) / x[1]) - norm.cdf((a - x[0]) / x[1])) - cumprob_likely
        )
        return root1**2 + root2**2

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - a},  
        {'type': 'ineq', 'fun': lambda x: x[1]}  
        ]    
    est = minimize(fn, [max(likely2025_mean, a + 1), 1.0], constraints=constraints)
    return {"mean": est.x[0], "sd": est.x[1]}


def sample_param(N):
    a_samples = np.random.uniform(operational_min, operational_max, N)
    b_samples = np.random.uniform(possible2025_min, possible2025_max, N)
    cap_1_samples = np.random.uniform(likely2025_min, likely2025_max, N)
    return a_samples, b_samples, cap_1_samples


def stochastic_samples(a_samples, b_samples, cap_1_samples, cumprob_likely, likely2025_mean, N):
    start_samples_all = []
    growth_rate_samples_all = []

    for i in range(N):
        a = a_samples[i]
        b = b_samples[i]
        cap_1 = cap_1_samples[i]

        dist_params = truncated_normal(a, b, cap_1, cumprob_likely, likely2025_mean)

        if dist_params["sd"] <= 0:
            dist_params["sd"] = 1
        
        start_samples = truncnorm.rvs(
            (a - dist_params["mean"]) / dist_params["sd"],
            (b - dist_params["mean"]) / dist_params["sd"],
            loc=dist_params["mean"],
            scale=dist_params["sd"],
            size=1
        )

        growth_rate_samples = truncnorm.rvs(
            a_std_growth,
            b_growth,
            loc=mean_growth,
            scale=std_growth,
            size=1
        )

        start_samples_all.append(start_samples[0])
        growth_rate_samples_all.append(growth_rate_samples[0])

    return np.array(start_samples_all), np.array(growth_rate_samples_all)


def diffusion(demand_jet_fuel, start_samples, growth_rate_samples, delta_t, start_year, end_year):
    results = []
    for i in range(len(start_samples)):
        start = start_samples[i]
        growth_rate = growth_rate_samples[i]
        years = np.arange(start_year, end_year + delta_t, delta_t)
        
        interpolator = interp1d(demand_jet_fuel["year"], demand_jet_fuel["demand"], kind="linear", fill_value="extrapolate")
        fine_demand = interpolator(years)
        
        adjusted_growth = (1 + growth_rate)**delta_t - 1
        
        forecast = [start]
        for t in range(1, len(fine_demand)):
            prev_forecast = forecast[-1]
            current_demand = fine_demand[t]
            new_forecast = prev_forecast + adjusted_growth * (1 - prev_forecast / current_demand) * prev_forecast
            forecast.append(new_forecast)
        
        results.append(pd.DataFrame({"year": years, "demand": fine_demand, "forecast": forecast}))
    
    return results

#Defining parameters
N = 50000  # Number of Monte Carlo simulations

status_capacity = "Announced_capacity_by_status_EU.csv"
status_capacity = pd.read_csv(status_capacity)
status_capacity.columns = [col.strip() for col in status_capacity.columns]
operational_min = status_capacity.loc[status_capacity.iloc[:, 0] <= 2024, "Operational_low"].sum()
operational_max = status_capacity.loc[status_capacity.iloc[:, 0] <= 2024, "Operational_high"].sum()
possible2025_min = (
    operational_min +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_low"].sum() +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_low"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FEED_low"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Feasibility/Permitting_low"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Concept_low"].sum()
)
possible2025_max = (
    operational_max +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_high"].sum() +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_high"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FEED_high"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Feasibility/Permitting_high"].sum()+
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Concept_high"].sum()
)

likely2025_min = (
    operational_min +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_low"].sum() +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_low"].sum()
)

likely2025_max = (
    operational_max +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_high"].sum() +
    status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_high"].sum()
)

cumprob_likely = 1

likely2025_mean = ((operational_min+operational_max)/2 + 
                   ((status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_low"].sum()+
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Under construction_high"].sum())/2 +
                   (status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_low"].sum()+
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FID_high"].sum())/2 +
                   (status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FEED_low"].sum()+
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "FEED_high"].sum())/2+
                   (status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Feasibility/Permitting_low"].sum()+
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Feasibility/Permitting_high"].sum())/2+
                   (status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Concept_low"].sum()+
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Concept_high"].sum())/2)*0.26)



delta_t = 1  
start_year = 2025
end_year = 2050
#Demand Anticipation. For sensitivity, change this value.
demand_anticipation = 1 #anticipation years of policy targets. For sensitivity: 0 and 2
a_samples, b_samples, cap_1_samples = sample_param(N)


demand_jet_fuel = pd.DataFrame({
    "year": [2025,2030-demand_anticipation,2050-demand_anticipation,2050],
    "demand": [1,2.8,35.7,35.7]
})

start_samples, growth_rate_samples = stochastic_samples(
    a_samples, b_samples, cap_1_samples, cumprob_likely, likely2025_mean, N
)


simulation_results = diffusion(demand_jet_fuel, start_samples, growth_rate_samples, delta_t, start_year, end_year)

final_results = pd.concat(simulation_results, keys=range(len(simulation_results)), names=["sample"])
final_results.reset_index(level="sample", inplace=True)

# Show results including IQR across all samples
def results_with_iqr(final_results):
    stats = (
        final_results
        .groupby("year")["forecast"]
        .agg(
            median="median",
            p5=lambda x: np.percentile(x, 5),
            p25=lambda x: np.percentile(x, 25),
            p50=lambda x: np.percentile(x, 50),
            p75=lambda x: np.percentile(x, 75),
            p95=lambda x: np.percentile(x, 95),
        )
        .reset_index()
    )
    stats["IQR"] = stats["p75"] - stats["p25"]
    return stats

stats_with_iqr = results_with_iqr(final_results)

median_demand = (
    final_results
    .groupby("year")["demand"]
    .median()
    .reset_index()
    .rename(columns={"demand": "median_demand"})
)



stats_with_iqr = stats_with_iqr.merge(median_demand, on="year", how="left")
quantiles = np.arange(0.05, 1.0, 0.05) 
stats_with_iqr = final_results.groupby("year")["forecast"].quantile(quantiles).unstack()
stats_with_iqr.columns = [f"p{int(q * 100)}" for q in quantiles]
stats_with_iqr = stats_with_iqr.reset_index()
stats_with_iqr = stats_with_iqr.merge(median_demand, on="year", how="left")
stats_with_iqr.rename(columns={"median_demand": "demand"}, inplace=True)

