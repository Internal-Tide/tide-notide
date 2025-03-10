# %%
import proplot as pplt
import numpy as np
import xarray as xr
import cmaps
import warnings

warnings.filterwarnings("ignore")
# %%
ds_ss_t = xr.open_dataset("/data08/tianzw/mean/tide/ss-r.nc")
ds_ss_n = xr.open_dataset("/data08/tianzw/mean/notide/ss-r.nc")
ds_tt_t = xr.open_dataset("/data08/tianzw/mean/tide/tt-r.nc")
ds_tt_n = xr.open_dataset("/data08/tianzw/mean/notide/tt-r.nc")
ds_uu_t = xr.open_dataset("/data08/tianzw/mean/tide/uu-r.nc")
ds_uu_n = xr.open_dataset("/data08/tianzw/mean/notide/uu-r.nc")
ds_vv_t = xr.open_dataset("/data08/tianzw/mean/tide/vv-r.nc")
ds_vv_n = xr.open_dataset("/data08/tianzw/mean/notide/vv-r.nc")
ds_woa_tt = xr.open_dataset(
    "/data08/tianzw/woa/0.25du/annual/temperature/woa18_temp-vh.nc", decode_times=False
)
ds_woa_ss = xr.open_dataset(
    "/data08/tianzw/woa/0.25du/annual/salinity/woa18_sality-vh.nc", decode_times=False
)
ds_elevation = xr.open_dataset("/data08/tianzw/elevation.nc")
ss_t = ds_ss_t["ss"].values[0, :, :, :]
ss_n = ds_ss_n["ss"].values[0, :, :, :]
tt_t = ds_tt_t["tt"].values[0, :, :, :]
tt_n = ds_tt_n["tt"].values[0, :, :, :]
woa_tt = ds_woa_tt["t_an"].values[0, :, :, :]
woa_ss = ds_woa_ss["s_an"].values[0, :, :, :]
elevation = ds_elevation["z0hour"][:, :]
lev = ds_ss_t["lev1"].values
u_t = np.sqrt(
    ds_uu_t["uu"].values[0, :, :, :] ** 2 + ds_vv_t["vv"].values[0, :, :, :] ** 2
)
u_n = np.sqrt(
    ds_uu_n["uu"].values[0, :, :, :] ** 2 + ds_vv_n["vv"].values[0, :, :, :] ** 2
)
lon = ds_ss_t.lon.values
lat = ds_ss_t.lat.values


# %%
def mean_elevation(array, elevation):
    r = np.unique(elevation)
    r = np.delete(r, np.where(np.isnan(r)))
    r = np.insert(r, 0, [5, 10, 15, 20])
    arr = array.copy()
    arr1 = np.zeros((55, 720, 1440))
    results = np.zeros((55, 55))
    arr1[[0, 1, 2, 3], :, :] = 1
    for l in range(4, 55):
        arr1[l, :, :] = np.where(elevation == r[l], 1, np.nan)
    for k in range(55):
        for l in range(55):
            if k > l:
                results[k, l] = np.nan
            else:
                results[k, l] = np.nanmean(arr[k, :, :] * arr1[l, :, :])

    return results


# %%
ss_t_mean = mean_elevation(ss_t, elevation)
ss_n_mean = mean_elevation(ss_n, elevation)
# %%
tt_t_mean = mean_elevation(tt_t, elevation)
tt_n_mean = mean_elevation(tt_n, elevation)
# %%
woa_tt_mean = mean_elevation(woa_tt, elevation)
woa_ss_mean = mean_elevation(woa_ss, elevation)
# %%
u_t_mean = mean_elevation(u_t, elevation)
u_n_mean = mean_elevation(u_n, elevation)
# %%
a = mean_elevation(ss_t - ss_n, elevation)
b = mean_elevation(ss_t - woa_ss, elevation)
c = mean_elevation(ss_n - woa_ss, elevation)
# %%
d = mean_elevation(tt_t - tt_n, elevation)
e = mean_elevation(tt_t - woa_tt, elevation)
f = mean_elevation(tt_n - woa_tt, elevation)


# %%
fig, axs = pplt.subplots(nrows=1, ncols=3, axwidth=6, yreverse=True, xreverse=True)
axs.format(
    labels=True,
    abc=False,
    abcloc="l",
    land=True,
)
axs[2].contourf(
    lev,
    lev,
    (u_t_mean - u_n_mean) * 100,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.3, 0.3, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(cm/s)",
        "length": 0.9,
        "ticks": np.linspace(-0.3, 0.3, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0].contourf(
    lev,
    lev,
    (u_t_mean) * 100,
    cmap="jet",
    levels=np.linspace(0, 4, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(cm/s)",
        "length": 0.9,
        "ticks": np.linspace(0, 4, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1].contourf(
    lev,
    lev,
    (u_n_mean) * 100,
    cmap="jet",
    levels=np.linspace(0, 4, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(cm/s)",
        "length": 0.9,
        "ticks": np.linspace(0, 4, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)

axs[0].format(title="Exp_tide")
axs[1].format(title="Exp_ctl")
axs[2].format(title="Exp_tide-Exp_ctl")
axs.format(
    xlabel="Bathymetry (m)",
    ylabel="Depth (m)",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Velocity",
    suptitlesize=30,
)
fig.savefig("./pics/velocity.png")

# %%
fig, axs = pplt.subplots(nrows=2, ncols=3, axwidth=6, yreverse=True, xreverse=True)
axs.format(
    labels=True,
    abc=True,
    abcloc="l",
)
axs[1, 0].contourf(
    lev,
    lev,
    a,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.05, 0.05, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.05, 0.05, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 1].contourf(
    lev,
    lev,
    b,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.2, 0.2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.2, 0.2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 2].contourf(
    lev,
    lev,
    c,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.2, 0.2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.2, 0.2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].contourf(
    lev,
    lev,
    ss_t_mean,
    cmap="jet",
    levels=np.linspace(34, 36, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(34, 36, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 1].contourf(
    lev,
    lev,
    ss_n_mean,
    cmap="jet",
    levels=np.linspace(34, 36, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(34, 36, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 2].contourf(
    lev,
    lev,
    woa_ss_mean,
    cmap="jet",
    levels=np.linspace(34, 36, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(34, 36, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 0].format(title="Exp_tide-Exp_ctl")
axs[1, 1].format(title="Exp_tide-WOA18")
axs[1, 2].format(title="Exp_ctl-WOA18")
axs[0, 0].format(title="Exp_tide")
axs[0, 1].format(title="Exp_ctl")
axs[0, 2].format(title="WOA18")

axs.format(
    xlabel="Bathymetry (m)",
    ylabel="Depth (m)",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Salinity",
    suptitlesize=30,
)
fig.savefig("./pics/salitity.png")
# %%
fig, axs = pplt.subplots(nrows=2, ncols=3, axwidth=6, yreverse=True, xreverse=True)
axs.format(
    labels=True,
    abc=True,
    abcloc="l",
)
axs[1, 0].contourf(
    lev,
    lev,
    d,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.5, 0.5, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-0.5, 0.5, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 1].contourf(
    lev,
    lev,
    e,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 2].contourf(
    lev,
    lev,
    f,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].contourf(
    lev,
    lev,
    tt_t_mean,
    cmap="jet",
    levels=np.linspace(-1, 20, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 20, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 1].contourf(
    lev,
    lev,
    tt_n_mean,
    cmap="jet",
    levels=np.linspace(-1, 20, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 20, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 2].contourf(
    lev,
    lev,
    woa_tt_mean,
    cmap="jet",
    levels=np.linspace(-1, 20, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 20, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 0].format(title="Exp_tide-Exp_ctl")
axs[1, 1].format(title="Exp_tide-WOA18")
axs[1, 2].format(title="Exp_ctl-WOA18")
axs[0, 0].format(title="Exp_tide")
axs[0, 1].format(title="Exp_ctl")
axs[0, 2].format(title="WOA18")
axs.format(
    xlabel="Bathymetry (m)",
    ylabel="Depth (m)",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Temperature",
    suptitlesize=30,
)
fig.savefig("./pics/temp.png")

# %%
ds_mld_t = xr.open_dataset("/data08/tianzw/mean/tide/mld-r.nc")
ds_mld_n = xr.open_dataset("/data08/tianzw/mean/notide/mld-r.nc")
mld_t = ds_mld_t["mld"][0, 0, :, :]
mld_n = ds_mld_n["mld"][0, 0, :, :]

proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(
    nrows=3, ncols=1, axwidth=6, yreverse=True, xreverse=True, proj=proj
)

m = axs[0].contourf(
    lon,
    lat,
    mld_t,
    cmap=cmaps.WhiteYellowOrangeRed,
    levels=np.linspace(10, 200, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": [10, 40, 70, 100, 130, 160, 190],
    },
    extend="both",
)
axs[1].contourf(
    lon,
    lat,
    mld_n,
    cmap=cmaps.WhiteYellowOrangeRed,
    levels=np.linspace(10, 200, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": [10, 40, 70, 100, 130, 160, 190],
    },
    extend="both",
)
axs[2].contourf(
    lon,
    lat,
    mld_t - mld_n,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-30, 30, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": [-30, -20, -10, 0, 10, 20, 30],
    },
    extend="both",
)
axs[0].format(title="Exp_tide")
axs[1].format(title="Exp_ctl")
axs[2].format(title="Exp_tide-Exp_ctl")

axs.format(
    coast=True,
    labels=True,
    land=True,
    landcolor="grey",
    abcloc="l",
    ticklabelsize=14,
    titleloc="l",
    titlesize=14,
    xlabelsize=14,
    ylabelsize=14,
    abcsize=14,
    abc="a.",
    suptitle="Mixed Layer Depth",
    suptitlesize=16,
)

fig.savefig("./pics/mld.png")


# %%
def vertical_mean(array):
    mask = np.where(np.isnan(array[:, :, :]), np.nan, 1)
    weight = np.cos(lat * np.pi / 180).reshape(720, 1)
    result = np.zeros(55)
    for k in range(55):
        result[k] = np.nansum(array[k, :, :] * weight) / np.nansum(
            weight * mask[k, :, :]
        )
    return result


# %%
result1 = vertical_mean(ss_t - woa_ss)
result2 = vertical_mean(ss_n - woa_ss)
result3 = vertical_mean(tt_t - woa_tt)
result4 = vertical_mean(tt_n - woa_tt)
# %%
fig, axs = pplt.subplots(
    nrows=1,
    ncols=2,
    axwidth=6,
    yreverse=True,
    xreverse=True,
    sharey=True,
    sharex=False,
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
)
axs[0].line(result1, lev, label="tide")
axs[0].line(result2, lev, label="ctl")
axs[0].vlines(0, 0, 5500, linestyle="--", color="k")
axs[0].format(
    xlim=(-0.1, 0.1),
    xlabel="(psu)",
    ylabel="Depth (m)",
    title="Salinity Bias",
)

axs[1].line(result3, lev, label="tide")
axs[1].line(result4, lev, label="ctl")
axs[1].vlines(0, 0, 5500, linestyle="--", color="k")
axs[1].legend(prop={'size': 22},ncol=1,loc=4)

axs[1].format(
    xlim=(-1, 1), xlabel="(°C)", ylabel="Depth (m)", title="Temperature Bias",
    
)

fig.savefig("./pics/vertical_mean.png")

# %%
ds_z0_t = xr.open_dataset("/data08/tianzw/mean/tide/z0-r.nc")
ds_z0_n = xr.open_dataset("/data08/tianzw/mean/notide/z0-r.nc")
ds_z0_aviso = xr.open_dataset("/data08/tianzw/aviso/sealevel_mr.nc")
z0_t = ds_z0_t["ssh"][0, 0, :, :].values
z0_n = ds_z0_n["ssh"][0, 0, :, :].values
z0_aviso = ds_z0_aviso["adt"][0, :, :].values
# %%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(nrows=3, ncols=2, axwidth=6, proj=proj)

m = axs[0, 1].contourf(
    lon,
    lat,
    z0_t - z0_aviso + 0.4189,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.6, 0.6, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-0.6, 0.6, 9, endpoint=True),
    },
    extend="both",
)
axs[1, 1].contourf(
    lon,
    lat,
    z0_n - z0_aviso + 0.4189,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.6, 0.6, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-0.6, 0.6, 9, endpoint=True),
    },
    extend="both",
)
axs[2, 1].contourf(
    lon,
    lat,
    z0_t - z0_n,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.3, 0.3, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-0.3, 0.3, 7, endpoint=True),
    },
    extend="both",
)
axs[0, 0].contourf(
    lon,
    lat,
    z0_t,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1.6, 1.6, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1.6, 1.6, 7, endpoint=True),
    },
    extend="both",
)
axs[1, 0].contourf(
    lon,
    lat,
    z0_n,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1.6, 1.6, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1.6, 1.6, 7, endpoint=True),
    },
    extend="both",
)
axs[2, 0].contourf(
    lon,
    lat,
    z0_aviso - 0.4189,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1.6, 1.6, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1.6, 1.6, 7, endpoint=True),
    },
    extend="both",
)
axs[0, 0].format(title="Exp_tide")
axs[1, 0].format(title="Exp_ctl")
axs[2, 0].format(title="Aviso")
axs[0, 1].format(title="Exp_tide - Aviso")
axs[1, 1].format(title="Exp_ctl - Aviso")
axs[2, 1].format(title="Exp_tide - Exp_ctl")
axs.format(
    coast=True,
    labels=True,
    land=True,
    landcolor="grey",
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="SSH",
    suptitlesize=30,
)
fig.savefig("./pics/z0.png")


# %%
def rmse(array1, array2):
    weight = np.cos(lat * np.pi / 180).reshape(720, 1)
    mask = np.where(np.isnan(array1), np.nan, 1)
    result = np.sqrt(
        np.nansum((array1 - array2) ** 2 * weight) / np.nansum(mask * weight)
    )
    return result


# %%
rmse(z0_t, z0_aviso - 0.4189)
rmse(z0_n, z0_aviso - 0.4189)
aviso_low = xr.where(
    (ds_z0_aviso["lat"] < 30) & (ds_z0_aviso["lat"] > -30),
    ds_z0_aviso["adt"] - 0.4189,
    np.nan,
)[:, 0, :].values
# %%
ds_tt_tide_g = xr.open_dataset("/data08/tianzw/mean/tide/tt.nc", decode_times=False)
ds_tt_notide_g = xr.open_dataset("/data08/tianzw/mean/notide/tt.nc", decode_times=False)
ds_ss_tide_g = xr.open_dataset("/data08/tianzw/mean/tide/ss.nc", decode_times=False)
ds_ss_notide_g = xr.open_dataset("/data08/tianzw/mean/notide/ss.nc", decode_times=False)
ds_basin = xr.open_dataset(
    "/data08/tianzw/moc/basin_code/LICOM_Basin_3600X2302X55_tripole_omip_20170529.DATA.nc",
    decode_times=False,
)
ds_basin = ds_basin.rename({"lon": "x", "lat": "y"})
ds_basin = ds_basin.rename({"ulon": "lon", "ulat": "lat"})

# %%
tt_tide_atl = ds_tt_tide_g.where(
    (ds_basin.basin.data == 1) | (ds_basin.basin.data == 2)
)
tt_notide_atl = ds_tt_notide_g.where(
    (ds_basin.basin.data == 1) | (ds_basin.basin.data == 2)
)
ss_tide_atl = ds_ss_tide_g.where(
    (ds_basin.basin.data == 1) | (ds_basin.basin.data == 2)
)
ss_notide_atl = ds_ss_notide_g.where(
    (ds_basin.basin.data == 1) | (ds_basin.basin.data == 2)
)

# %%
tt_profile = (tt_tide_atl.tt[0, :, :, :] - tt_notide_atl.tt[0, :, :, :]).mean("x")
ss_profile = (ss_tide_atl.ss[0, :, :, :] - ss_notide_atl.ss[0, :, :, :]).mean("x")

# %%
lat_a = np.linspace(89.75, -89.75, 2302, endpoint=True)
# %%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(
    nrows=1,
    ncols=2,
    axwidth=6,
    yreverse=True,
    xreverse=False,
    sharey=True,
    sharex=False,
)

m = axs[0].contourf(
    lat_a[::-1][766:],
    tt_tide_atl.lev1.data,
    tt_profile[:, 766:],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0].contour(
    lat_a[::-1][766:],
    tt_tide_atl.lev1.data,
    tt_profile[:, 766:],
    levels=[0],
    color="k",
    extend="both",
)
axs[1].contourf(
    lat_a[::-1][766:],
    tt_tide_atl.lev1.data,
    ss_profile[:, 766:],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.3, 0.3, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.3, 0.3, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1].contour(
    lat_a[::-1][766:],
    tt_tide_atl.lev1.data,
    ss_profile[:, 766:],
    levels=[0],
    color="k",
    extend="both",
)

axs.format(
    labels=True,
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Atlantic zonal mean",
    suptitlesize=30,
)
axs[0].format(title="Exp_tide-Exp_ctl Salinity", ylabel="Depth(m)", xlabel="Latitude")
axs[1].format(title="Exp_tide-Exp_ctl Temperature", xlabel="Latitude")
fig.savefig("/home/tianzw/code/bigpaper/pics/ts-na.png")
# %%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(nrows=3, ncols=2, axwidth=6, proj=proj)

m = axs[0, 1].contourf(
    lon,
    lat,
    tt_t[0, :, :] - woa_tt[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-2, 2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-2, 2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 1].contourf(
    lon,
    lat,
    tt_n[0, :, :] - woa_tt[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-2, 2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-2, 2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2, 1].contourf(
    lon,
    lat,
    tt_t[0, :, :] - tt_n[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].contourf(
    lon,
    lat,
    tt_t[0,:,:],
    cmap="jet",
    levels=np.linspace(-2, 32, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-2, 32, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 0].contourf(
    lon,
    lat,
    tt_n[0,:,:],
    cmap="jet",
    levels=np.linspace(-2, 32, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-2, 32, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2, 0].contourf(
    lon,
    lat,
    woa_tt[0,:,:],
    cmap="jet",
    levels=np.linspace(-2, 32, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-2, 32, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].format(title="Exp_tide")
axs[1, 0].format(title="Exp_ctl")
axs[2, 0].format(title="WOA18")
axs[0, 1].format(title="Exp_tide - WOA18")
axs[1, 1].format(title="Exp_ctl - WOA18")
axs[2, 1].format(title="Exp_tide - Exp_ctl")
axs.format(
    coast=True,
    labels=True,
    land=True,
    landcolor="grey",
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="SST",
    suptitlesize=30,
)
fig.savefig("./pics/sst.png")
#%%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(nrows=3, ncols=2, axwidth=6, proj=proj)

m = axs[0, 1].contourf(
    lon,
    lat,
    ss_t[0, :, :] - woa_ss[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 1].contourf(
    lon,
    lat,
    ss_n[0, :, :] - woa_ss[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-1, 1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-1, 1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2, 1].contourf(
    lon,
    lat,
    ss_t[0, :, :] - ss_n[0, :, :],
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.5, 0.5, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(-0.5, 0.5, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].contourf(
    lon,
    lat,
    ss_t[0,:,:],
    cmap="jet",
    levels=np.linspace(30, 38, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(30, 38, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1, 0].contourf(
    lon,
    lat,
    ss_n[0,:,:],
    cmap="jet",
    levels=np.linspace(30, 38, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(30, 38, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2, 0].contourf(
    lon,
    lat,
    woa_ss[0,:,:],
    cmap="jet",
    levels=np.linspace(30, 38, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(m)",
        "length": 0.9,
        "ticks": np.linspace(30, 38, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0, 0].format(title="Exp_tide")
axs[1, 0].format(title="Exp_ctl")
axs[2, 0].format(title="WOA18")
axs[0, 1].format(title="Exp_tide - WOA18")
axs[1, 1].format(title="Exp_ctl - WOA18")
axs[2, 1].format(title="Exp_tide - Exp_ctl")
axs.format(
    coast=True,
    labels=True,
    land=True,
    landcolor="grey",
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="SSS",
    suptitlesize=30,
)
fig.savefig("./pics/sss.png")
#%%
ds_temp_n_v = ds_tt_n["tt"][0,:,:,:]
ds_temp_t_v = ds_tt_t["tt"][0,:,:,:]
ds_temp_woa_v = ds_woa_tt["t_an"][0,:,:,:]
#%%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(
    nrows=1,
    ncols=3,
    axwidth=6,
    yreverse=True,
    xreverse=False,
    sharey=True,
    sharex=True,
)

axs[0].contourf(
    lat,
    ds_tt_n["lev1"],
    ds_temp_t_v.mean("lon").data-ds_temp_woa_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-2, 2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-2, 2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0].contour(    
    lat,
    ds_tt_n["lev1"],
    ds_temp_t_v.mean("lon").data-ds_temp_woa_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs[1].contourf(
    lat,
    ds_tt_n["lev1"],
    ds_temp_n_v.mean("lon").data-ds_temp_woa_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-2, 2, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-2, 2, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1].contour(    
    lat,
    ds_tt_n["lev1"],
    ds_temp_n_v.mean("lon").data-ds_temp_woa_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs[2].contourf(
    lat,
    ds_tt_n["lev1"],
    ds_temp_t_v.mean("lon").data-ds_temp_n_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.5, 0.5, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(°C)",
        "length": 0.9,
        "ticks": np.linspace(-0.5, 0.5, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2].contour(    
    lat,
    ds_tt_n["lev1"],
    ds_temp_t_v.mean("lon").data-ds_temp_n_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs.format(
    labels=True,
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Zonal mean temperature",
    suptitlesize=30,
)
axs[0].format(title="Exp_tide-WOA18", ylabel="Depth(m)",xlabel="Latitude")
axs[1].format(title="Exp_ctl-WOA18", ylabel="Depth(m)",xlabel="Latitude")
axs[2].format(title="Exp_tide-Exp_ctl", ylabel="Depth(m)",xlabel="Latitude")

fig.savefig("/home/tianzw/code/bigpaper/pics/zonal-tt.png")
#%%
ds_ss_n_v = ds_ss_n["ss"][0,:,:,:]
ds_ss_t_v = ds_ss_t["ss"][0,:,:,:]
ds_ss_woa_v = ds_woa_ss["s_an"][0,:,:,:]
#%%
proj = pplt.Proj("cyl", lon_0=180)
fig, axs = pplt.subplots(
    nrows=1,
    ncols=3,
    axwidth=6,
    yreverse=True,
    xreverse=False,
    sharey=True,
    sharex=True,
)

axs[0].contourf(
    lat,
    ds_ss_n["lev1"],
    ds_ss_t_v.mean("lon").data-ds_ss_woa_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.3, 0.3, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.3, 0.3, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[0].contour(    
    lat,
    ds_tt_n["lev1"],
    ds_ss_t_v.mean("lon").data-ds_ss_woa_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs[1].contourf(
    lat,
    ds_ss_n["lev1"],
    ds_ss_n_v.mean("lon").data-ds_ss_woa_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.3, 0.3, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.3, 0.3, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[1].contour(    
    lat,
    ds_ss_n["lev1"],
    ds_ss_n_v.mean("lon").data-ds_ss_woa_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs[2].contourf(
    lat,
    ds_ss_n["lev1"],
    ds_ss_t_v.mean("lon").data-ds_ss_n_v.mean("lon").data,
    cmap=cmaps.NCV_blu_red,
    levels=np.linspace(-0.1, 0.1, 51, endpoint=True),
    colorbar="r",
    colorbar_kw={
        "label": "(psu)",
        "length": 0.9,
        "ticks": np.linspace(-0.1, 0.1, 9, endpoint=True),
        "ticklabelsize":22,
        "labelsize":22
    },
    extend="both",
)
axs[2].contour(    
    lat,
    ds_ss_n["lev1"],
    ds_ss_t_v.mean("lon").data-ds_ss_n_v.mean("lon").data,
    levels=[0],
    color="k",
    extend="both")
axs.format(
    labels=True,
    abcloc="l",
    ticklabelsize=26,
    titleloc="l",
    titlesize=26,
    xlabelsize=26,
    ylabelsize=26,
    abcsize=26,
    abc="a.",
    suptitle="Zonal mean salinity",
    suptitlesize=30,
)
axs[0].format(title="Exp_tide-WOA18", ylabel="Depth(m)",xlabel="Latitude")
axs[1].format(title="Exp_ctl-WOA18", ylabel="Depth(m)",xlabel="Latitude")
axs[2].format(title="Exp_tide-Exp_ctl", ylabel="Depth(m)",xlabel="Latitude")

fig.savefig("/home/tianzw/code/bigpaper/pics/zonal-ss.png")