import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import curve_fit

@dataclass
class BeamProperties:
    """Holds the calculated properties of a single beam image."""
    vertical_beam_size: float
    horizontal_beam_size: float
    total_intensity: float
    peak_intensity: float
    vertical_line_profile: np.ndarray
    roi_image: np.ndarray
    fwhm: float
    fitted_gaussian_profile: Optional[np.ndarray] = None
    gaussian_fit_parameters: Optional[Dict[str, float]] = None
    crop_metadata: Optional[Dict[str, Any]] = None


class BeamAnalyzer:
    def __init__(self, image: np.ndarray, crop: bool=True, pixel_size: float=1.0):
        """
        Parameters
        ----------
        image : np.ndarray
            2D grayscale image of the beam.
        crop : bool
            Whether to crop to a region of interest.
        pixel_size : float
            Physical size of one pixel (µm, mm, etc.). Default=1.0 means pixel units.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input must be a 2D NumPy array.")
        self.raw_image = image
        self.roi_image: Optional[np.ndarray] = None
        self.crop_meta: Optional[Dict[str, Any]] = None
        self._crop = crop
        self.pixel_size = pixel_size

    def analyze(self, profile_width: int = 5) -> Optional[BeamProperties]:
        if self._crop:
            if not self._crop_to_roi():
                return None
        else:
            self.roi_image = self.raw_image

        if self.roi_image is None or self.roi_image.sum() == 0:
            return None

        v_beam_size = self._calculate_beam_size(axis=1) * self.pixel_size
        h_beam_size = self._calculate_beam_size(axis=0) * self.pixel_size
        total_intensity = float(np.sum(self.roi_image))
        peak_intensity = float(np.max(self.roi_image))

        line_profile, fitted_gaussian, fit_params = self._extract_and_fit_profile(profile_width)

        # Prefer Gaussian-fit FWHM if available
        if fit_params and "sigma" in fit_params:
            fwhm = 2.355 * fit_params["sigma"] * self.pixel_size
        else:
            fwhm = self._calculate_fwhm_from_profile(line_profile) * self.pixel_size

        if line_profile is None:
            return None

        return BeamProperties(
            vertical_beam_size=v_beam_size,
            horizontal_beam_size=h_beam_size,
            total_intensity=total_intensity,
            peak_intensity=peak_intensity,
            vertical_line_profile=line_profile,
            fitted_gaussian_profile=fitted_gaussian,
            gaussian_fit_parameters=fit_params,
            roi_image=self.roi_image,
            crop_metadata=self.crop_meta,
            fwhm=fwhm
        )

    def _crop_to_roi(self) -> bool:
        top, bottom = 80, 260
        left, right = 1730, 1780

        if self.raw_image.shape[0] < bottom or self.raw_image.shape[1] < right:
            logging.error(f"Image shape {self.raw_image.shape} is unsuitable for the crop window.")
            return False

        self.roi_image = self.raw_image[top:bottom, left:right]
        self.crop_meta = {
            'top': top, 'bottom': bottom, 'left': left, 'right': right,
            'original_shape': self.raw_image.shape
        }
        return True

    def _calculate_beam_size(self, axis: int) -> float:
        """
        Calculate 4σ beam size along a given axis (0=rows/horizontal, 1=columns/vertical).
        """
        if self.roi_image is None or self.roi_image.sum() == 0:
            return 0.0

        profile = np.sum(self.roi_image, axis=axis)
        total_intensity = np.sum(profile)
        coords = np.arange(len(profile))
        mean_val = np.sum(coords * profile) / total_intensity
        variance = np.sum(profile * (coords - mean_val)**2) / total_intensity
        sigma = np.sqrt(variance)
        return 4 * sigma

    def _extract_and_fit_profile(self, profile_width: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]]]:
        if self.roi_image is None or self.roi_image.sum() == 0:
            return None, None, None

        horizontal_profile = np.sum(self.roi_image, axis=0)
        if horizontal_profile.sum() == 0:
            return None, None, None

        x_coords = np.arange(len(horizontal_profile))
        mean_x = np.sum(x_coords * horizontal_profile) / np.sum(horizontal_profile)
        center_col = int(round(mean_x))

        half_width = profile_width // 2
        start_col = max(0, center_col - half_width)
        end_col = min(self.roi_image.shape[1], center_col + half_width + 1)

        line_profile = np.sum(self.roi_image[:, start_col:end_col], axis=1)
        fitted_gaussian, fit_params = self._fit_gaussian_to_profile(line_profile)
        return line_profile, fitted_gaussian, fit_params

    @staticmethod
    def _gaussian_function(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
        if sigma <= 0:
            sigma = 1e-6
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset

    @staticmethod
    def _fit_gaussian_to_profile(profile: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        if profile.sum() == 0:
            return None, None

        y_coords = np.arange(len(profile))
        amplitude_guess = np.max(profile) - np.min(profile)
        center_guess = np.argmax(profile)
        sigma_guess = np.sqrt(np.sum(profile * (y_coords - center_guess)**2) / np.sum(profile))
        offset_guess = np.min(profile)
        initial_guess = [amplitude_guess, center_guess, sigma_guess, offset_guess]

        try:
            popt, _ = curve_fit(BeamAnalyzer._gaussian_function, y_coords, profile, p0=initial_guess, maxfev=2000)
            fitted_gaussian = BeamAnalyzer._gaussian_function(y_coords, *popt)
            fit_params = {
                'amplitude': popt[0],
                'center': popt[1],
                'sigma': abs(popt[2]),
                'offset': popt[3],
                'fit_quality_r2': np.corrcoef(profile, fitted_gaussian)[0, 1]**2
            }
            return fitted_gaussian, fit_params
        except (RuntimeError, ValueError) as e:
            logging.warning(f"Gaussian fit failed: {e}")
            return None, None

    @staticmethod
    def _calculate_fwhm_from_profile(profile: np.ndarray) -> float:
        if profile is None or profile.size == 0:
            return 0.0
        try:
            max_val = np.max(profile)
            half_max = max_val / 2.0
            above_half_max_indices = np.where(profile > half_max)[0]
            if above_half_max_indices.size < 2:
                return 0.0
            return float(above_half_max_indices[-1] - above_half_max_indices[0])
        except (ValueError, IndexError) as e:
            logging.warning(f"FWHM calculation failed: {e}")
            return 0.0
