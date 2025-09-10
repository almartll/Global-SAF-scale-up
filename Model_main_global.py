import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize

LTAG_demand = 'ICAO LTAG Data to support States analysis.xlsx'
sheet_name = 'F2-High'
demand_2050 = pd.read_excel(LTAG_demand, sheet_name=sheet_name, header=None, usecols="B", skiprows=33, nrows=1).iloc[0, 0]

source_data = "Statistical Review of World Energy Data.xlsx" 
source_data_ethanol = "HATCH_1.0.xlsx"  

df_solar = pd.read_excel(source_data, sheet_name="Solar Installed Capacity", header=None)
df_wind = pd.read_excel(source_data, sheet_name="Wind Installed Capacity", header=None)
df_biofuels = pd.read_excel(source_data, sheet_name="Biofuels production - kboed", header=None)
df_ethanol = pd.read_excel(source_data_ethanol, sheet_name="HATCH_1.0", header=None)

solar_years = df_solar.iloc[3, 1:25].values  
wind_years = df_wind.iloc[3, 1:28].values  
biofuels_years = df_biofuels.iloc[2, 1:35].values  
ethanol_years = df_ethanol.iloc[0, 280:306].values  

solar_capacity = df_solar.iloc[77, 1:25].values  
wind_capacity = df_wind.iloc[69, 1:28].values  
biofuels_capacity = df_biofuels.iloc[45, 1:35].values  
ethanol_capacity = df_ethanol.iloc[74, 280:306].values  

def exp_growth_solar(x, a, b):
    return a * (1 + b) ** (x-solar_years[0]-1)

def exp_growth_wind(x, a, b):
    return a * (1 + b) ** (x-wind_years[0]-1)

def exp_growth_biofuels(x, a, b):
    return a * (1 + b) ** (x-biofuels_years[0]-1)

def exp_growth_ethanol(x, a, b):
    return a * (1 + b) ** (x-ethanol_years[0]-1)


time_slice_length = 5

results_solar = []
results_wind = []
results_biofuels = []
results_ethanol = []

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


selected_solar_years = [2000, 2001,2002,2003,2004,2005,2006,2007,2008,2009]  
solar_selected = df_results_solar[df_results_solar['slice_start'].isin(selected_solar_years)]
solar_growth_rates = np.array(solar_selected['growth_rate'].values)

selected_wind_years = [1997, 1998,1999,2000,2001,2002,2003,2004,2005,2006]  
wind_selected = df_results_wind[df_results_wind['slice_start'].isin(selected_wind_years)]
wind_growth_rates = np.array(wind_selected['growth_rate'].values)

selected_biofuel_years = [1999,2000,2001,2002,2003,2004,2005,2006,2007]  
biofuels_selected = df_results_biofuels[df_results_biofuels['slice_start'].isin(selected_biofuel_years)]
biofuels_growth_rates = np.array(biofuels_selected['growth_rate'].values)

selected_ethanol_years = [1999,2000,2001,2002,2003,2004,2005,2006,2007]  
ethanol_selected = df_results_ethanol[df_results_ethanol['slice_start'].isin(selected_ethanol_years)]
ethanol_growth_rates = np.array(ethanol_selected['growth_rate'].values)


# Combine growth rates
# FOR BIOFUELS/ETHANOL: substitute solar and wind with biofuels and ethanol and run the code 
combined_growth_rates = np.concatenate([solar_growth_rates,wind_growth_rates])


# Define parameters for the truncated normal distribution for growth rates
lowerbound_growth = np.min(combined_growth_rates)  
upperbound_growth = np.max(combined_growth_rates)  
mean_growth = np.mean(combined_growth_rates)  
std_growth = np.std(combined_growth_rates)  

lowerbound_std_growth = (lowerbound_growth - mean_growth) / std_growth
upperbound_std_growth = (upperbound_growth - mean_growth) / std_growth

def truncated_normal(lowerbound, upperbound, cap_1, cumprob_likely, mean):
    def fn(x):
        root1 = (
            x[0] + x[1] * 
            (-norm.pdf((upperbound - x[0]) / x[1]) + norm.pdf((lowerbound - x[0]) / x[1])) / 
            (norm.cdf((upperbound - x[0]) / x[1]) - norm.cdf((lowerbound - x[0]) / x[1])) - mean
        )
        root2 = (
            (norm.cdf((cap_1 - x[0]) / x[1]) - norm.cdf((lowerbound - x[0]) / x[1])) / 
            (norm.cdf((upperbound - x[0]) / x[1]) - norm.cdf((lowerbound - x[0]) / x[1])) - cumprob_likely
        )
        return root1**2 + root2**2

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - lowerbound},  
        {'type': 'ineq', 'fun': lambda x: x[1]}  
        ]      
    est = minimize(fn, [max(mean, lowerbound + 1), 1.0], constraints=constraints)
    return {"mean": est.x[0], "sd": est.x[1]}


def sample_param(N):
    operational_samples = np.random.uniform(operational_min, operational_max, N)
    possible_samples = np.random.uniform(possible2025_min, possible2025_max, N)
    likely_samples = np.random.uniform(likely2025_min, likely2025_max, N)
    return operational_samples, possible_samples, likely_samples

def stochastic_samples(operational_samples, possible_samples, likely_samples, cumprob_likely, likely2025_mean, N):
    capacity_samples_all = []
    growth_rate_samples_all = []

    for i in range(N):
        lowerbound_capacity = operational_samples[i]
        upperbound_capacity = possible_samples[i]
        cap_1 = likely_samples[i]

        dist_params = truncated_normal(lowerbound_capacity, upperbound_capacity, cap_1, cumprob_likely, likely2025_mean)

        initial_capacity_samples = truncnorm.rvs(
            (lowerbound_capacity - dist_params["mean"]) / dist_params["sd"],
            (upperbound_capacity - dist_params["mean"]) / dist_params["sd"],
            loc=dist_params["mean"],
            scale=dist_params["sd"],
            size=1
        )

        growth_rate_samples = truncnorm.rvs(
            lowerbound_std_growth,
            upperbound_std_growth,
            loc=mean_growth,
            scale=std_growth,
            size=1
        )

        capacity_samples_all.append(initial_capacity_samples[0])
        growth_rate_samples_all.append(growth_rate_samples[0])

    return np.array(capacity_samples_all), np.array(growth_rate_samples_all)


def diffusion(demand_jet_fuel, initial_capacity_samples, growth_rate_samples, delta_t, start_year, end_year):
    results = []
    for i in range(len(initial_capacity_samples)):
        initial_capacity = initial_capacity_samples[i]
        growth_rate = growth_rate_samples[i]
        years = np.arange(start_year, end_year + delta_t, delta_t)
        
        interpolator = interp1d(demand_jet_fuel["year"], demand_jet_fuel["demand"], kind="linear", fill_value="extrapolate")
        estimate_demand = interpolator(years)
        
        adjusted_growth = (1 + growth_rate)**delta_t - 1
        
        forecast = [initial_capacity]
        for t in range(1, len(estimate_demand)):
            prev_forecast = forecast[-1]
            current_demand = estimate_demand[t]
            new_forecast = prev_forecast + adjusted_growth * (1 - prev_forecast / current_demand) * prev_forecast
            forecast.append(new_forecast)
        
        results.append(pd.DataFrame({"year": years, "demand": estimate_demand, "forecast": forecast}))
    
    return results

#Defining parameters
N = 50000  # Number of Monte Carlo simulations

status_capacity = "Announced_capacity_by_status_global.csv"
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
                   status_capacity.loc[status_capacity.iloc[:, 0] <= 2025, "Concept_high"].sum())/2)*0.24)



delta_t = 1  
start_year = 2025
end_year = 2050

#Demand Anticipation. For sensitivity, change this value.
demand_anticipation = 1 #anticipation years of policy targets. For sensitivity: 0 and 2
operational_samples, possible_samples, likely_samples = sample_param(N)


demand_jet_fuel = pd.DataFrame({
    "year": [2025,2030-demand_anticipation, 2050-demand_anticipation,2050],
    "demand": [4.6,23,demand_2050/1000,demand_2050/1000]
})

initial_capacity_samples, growth_rate_samples = stochastic_samples(
    operational_samples, possible_samples, likely_samples, cumprob_likely, likely2025_mean, N
)
simulation_results = diffusion(demand_jet_fuel, initial_capacity_samples, growth_rate_samples, delta_t, start_year, end_year)

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



#CAPEX calculations with SAF Rules of Thumb (ICAO)
yearly_capex = pd.DataFrame({'Yearly Capacity Additions': stats_with_iqr['p50'].diff()})

#from https://www.icao.int/environmental-protection/Pages/SAF_RULESOFTHUMB.aspx
hefa_capex=(0.4+0.5)/2 
ptl_capex=(3.4+3.2)/2
ft_capex=(8.1+10.9+12.7)/3
atj_capex=(0.9+1.1+1.2+1.7)/4
other_capex=(5.9+6.2)/2

yearly_capex['Yearly CAPEX Investment in SAF Capacity (Billion$)'] = ((yearly_capex['Yearly Capacity Additions']*0.71*hefa_capex/0.0000000008)/1000000000+
                                        (yearly_capex['Yearly Capacity Additions']*0.112*atj_capex/0.0000000008)/1000000000+
                                        (yearly_capex['Yearly Capacity Additions']*0.09*ptl_capex/0.0000000008)/1000000000+
                                        (yearly_capex['Yearly Capacity Additions']*0.075*ft_capex/0.0000000008)/1000000000+
                                        (yearly_capex['Yearly Capacity Additions']*0.015*other_capex/0.0000000008)/1000000000)

yearly_capex['Cumulative CAPEX Investment (Billion$)'] = yearly_capex[
    'Yearly CAPEX Investment in SAF Capacity (Billion$)'
].cumsum()

yearly_capex['Cumulative CAPEX Investment in SAF Capacity needed (Billion$)'] = ((stats_with_iqr['demand']*0.71*hefa_capex/0.0000000008)/1000000000+
                                        (stats_with_iqr['demand']*0.112*atj_capex/0.0000000008)/1000000000+
                                        (stats_with_iqr['demand']*0.09*ptl_capex/0.0000000008)/1000000000+
                                        (stats_with_iqr['demand']*0.075*ft_capex/0.0000000008)/1000000000+
                                        (stats_with_iqr['demand']*0.015*other_capex/0.0000000008)/1000000000)



