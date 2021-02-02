# Utility code to convert the SMAL files for windows
Write-Host "Converting SMAL files"

function Convert-PKL {
    param (
        [string[]]$files
    )

    foreach ($f in $files) {
        Write-Host "Copying and converting file:" $f
        $win_file = $f.Replace(".pkl", "_WIN.pkl")
        Copy-Item $f -Destination $win_file
        (Get-Content $win_file -Raw).Replace("`r`n","`n") | Set-Content $win_file -Force
    }

}

$smalst_model_path = "data/SMALST/smpl_models"

$files = @(
    (Join-Path -Path $smalst_model_path "my_smpl_00781_4_all_template_w_tex_uv_001.pkl"),
    (Join-Path -Path $smalst_model_path "my_smpl_data_00781_4_all.pkl"),
    (Join-Path -Path $smalst_model_path "symIdx.pkl"),
    ("data/priors/walking_toy_symmetric_pose_prior_with_cov_35parts.pkl"))

# Write-Host $files
Convert-PKL($files)