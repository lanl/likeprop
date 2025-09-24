'''This module describes a wrapper class for probability density. 
We can use scipy.stats or tf.Distributions to build this'''
import abc
from typing import Generic, TypeVar, Protocol, Any
from numpy.typing import ArrayLike
# import scipy.stats


# Abstract base class for defining a PDF.
T = TypeVar('T')


class ProbabilityDensityFunction(Generic[T]):
    """Abstract PDF Wrapper Class for defining probability density functions

    Args:
        Generic (T): generic class to wrap around scipy.stats or tensorflow.Distributions
    """

    @abc.abstractmethod
    def relative_likelihood(self, point: T) -> Any:
        """returns the pdf evaluated at point (relative likelihood of a point)

        Args:
            point (T): a generic object

        Returns:
            float: pdf evaluated at point
        """

    @abc.abstractmethod
    def relative_log_likelihood(self, point: T) -> Any:
        """returns the log of pdf of point (relative log likelihood of a point)

        Args:
            point (T): a generic object

        Returns:
            float: log of the pdf evaluated at point
        """

#

# scipy.stats doesn't have a base density model to inherit so we will use the following
# protocol as an interface. Mainly for type checking

# pylint: disable=invalid-name


class ScipyPDF(Protocol):
    """A Generic scipy.stats based distribution class

    Args:
        Protocol (_type_): abstract class
    """

    def pdf(self, x) -> Any:
        """returns pdf of point

        Args:
            x (array_like): point to evalute pdf
        """

    def logpdf(self, x) -> Any:
        """returns log pdf of point

        Args:
            x (array_like): point to evalute pdf
        """

# pylint: enable=invalid-name


ScipyPdfT = TypeVar('ScipyPdfT', bound=ScipyPDF)


class DistributionFunction(ProbabilityDensityFunction[ArrayLike]):
    """A scipy.stats based PDF function on points

    Args:
        ProbabilityDensityFunction (_type_): _description_
    """

    def __init__(self, density_function: ScipyPDF):
        self._distribution = density_function

    def relative_likelihood(self, point: ArrayLike):
        return self._distribution.pdf(point)

    def relative_log_likelihood(self, point: ArrayLike):
        return self._distribution.logpdf(point)

    @property
    def distribution(self) -> ScipyPDF:
        """returns distribution function

        Returns:
            ScipyPdfT: scipy.stats distribution function
        """
        return self._distribution
