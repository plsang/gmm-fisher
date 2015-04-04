
#include "fisher.h"

template<class T>
fisher<T>::fisher( fisher_param &_param )
  : param(_param), gmm(0), iwgh(0), istd(0)
{
  ngrad = (int)param.grad_weights + (int)param.grad_means + (int)param.grad_variances;
  assert( (param.alpha>0.0) && (param.alpha<=1.0) ); 
}

template<class T>
fisher<T>::~fisher()
{
    gmm=0;

    delete[] iwgh;
    iwgh=0;

    delete[] istd;
    istd = 0;
  
    delete[] s0;
    for( int k=ngauss; k--; )
    {
        delete[] s1[k];
        delete[] s2[k];
    }
    delete [] s1;
    delete [] s2;
    s0 = 0;
    s1 = 0;
    s2 = 0;
}

template<class T>
void
fisher<T>::set_model( gaussian_mixture<T> &_gmm )
{
  gmm = &_gmm;
  ndim = gmm->n_dim();
  ngauss = gmm->n_gauss();

  fkdim = 0;
  if( param.grad_weights )
  {
    fkdim += ngauss;
  }
  if( param.grad_means )
  {
    fkdim += ngauss*ndim;
  }
  if( param.grad_variances )
  {
    fkdim += ngauss*ndim;
  }

  delete[] iwgh;

  // precompute inverse weights
  iwgh = new T[ngauss];
  for( int j=0; j<ngauss; ++j )
  {
    assert( gmm->coef[j]>0.0 );
    iwgh[j] = 1.0/gmm->coef[j];
  } 

  // precompute inverse standard deviations
  if( param.grad_means || param.grad_variances )
  {
    delete[] istd;
    istd = new T[ngauss*ndim];

    for( int j=0; j<ngauss; ++j ) 
    {
      T *var_j = gmm->var[j];
      T *istd_j = istd+j*ndim;
      for( int k=ndim; k--; ) 
      {
        assert( var_j[k]>0.0 );
        istd_j[k] = (T)1.0/(T)sqrtf( (float)var_j[k] );
      }
    }    
  }
  
  // Update Aug 6th, 2013: Set s0, s1, s2 here
  wghsum = 0.0;
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();
    
	s0 = new T[ngauss];
	memset( s0, 0, ngauss*sizeof(T));
	s1 = new T*[ngauss];
	for( int k=ngauss; k--; )
	{
	  s1[k] = new T[ndim];
	  memset( s1[k], 0, ndim*sizeof(T));
	}
	s2 = new T*[ngauss];
	for( int k=ngauss; k--; )
	{
	  s2[k] = new T[ndim];
	  memset( s2[k], 0, ndim*sizeof(T));
	}
}


template<class T>
int
fisher<T>::compute( std::vector<T*> &x, T *fk )
{
  std::vector<T> wghx( x.size(), 1.0 );  
  return compute( x, wghx, fk );
}

template<class T>
int
fisher<T>::compute( std::vector<T*> &x, T *fk, T *stats )
{
  std::vector<T> wghx( x.size(), 1.0 );  
  return compute( x, wghx, fk, stats );
}

template<class T>
int 
fisher<T>::accumulate( std::vector<T*> &x )
{  
    assert(gmm);
	 
	int nsamples = (int)x.size();
	
    wghsum += 1.0*nsamples;

	for( int i=0; i<nsamples; ++i )
    {
      gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances,
				  s0, s1, s2 );
    }
    
	return 0;
}

template<class T>
int 
fisher<T>::getfk(T *fk )
{  

  assert(gmm);

  assert( wghsum>0 );
  
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();

  T *p=fk;

  // Gradient w.r.t. the mixing weights
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
      p[j] = s0[j] / ( wghsum*(T)sqrtf((float)iwgh[j]) );
    } 
    p += ngauss;
  }

  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *mean_j = gmm->mean[j];
      T *istd_j = istd+j*ndim;
      T *p_j = p+j*ndim;
      T mc = (T)sqrtf((float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = mc * ( s1_j[k] - mean_j[k] * s0[j] ) * istd_j[k];
      }      
    }
    p += ngauss*ndim;     
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {

#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *s2_j = s2[j];
      T *mean_j = gmm->mean[j];
      T *var_j = gmm->var[j];
      T *p_j = p+j*ndim;
      T vc = (T)sqrtf(0.5f*(float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = vc * ( ( s2_j[k] + mean_j[k] * ( mean_j[k]*s0[j] - (T)2.0*s1_j[k] ) ) / var_j[k] - s0[j] );
      }   
    }
  } 
  
  alpha_and_lp_normalization(fk);
  
  return 0;
}

// get fk & save stats
template<class T>
int 
fisher<T>::getfk(T *fk, T*stats )
{  

  assert(gmm);

  assert( wghsum>0 );
  
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();

  T *p=fk;

  // Gradient w.r.t. the mixing weights
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
      p[j] = s0[j] / ( wghsum*(T)sqrtf((float)iwgh[j]) );
    } 
    p += ngauss;
  }

  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *mean_j = gmm->mean[j];
      T *istd_j = istd+j*ndim;
      T *p_j = p+j*ndim;
      T mc = (T)sqrtf((float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = mc * ( s1_j[k] - mean_j[k] * s0[j] ) * istd_j[k];
      }      
    }
    p += ngauss*ndim;     
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {

#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *s2_j = s2[j];
      T *mean_j = gmm->mean[j];
      T *var_j = gmm->var[j];
      T *p_j = p+j*ndim;
      T vc = (T)sqrtf(0.5f*(float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = vc * ( ( s2_j[k] + mean_j[k] * ( mean_j[k]*s0[j] - (T)2.0*s1_j[k] ) ) / var_j[k] - s0[j] );
      }   
    }
  } 
  
  alpha_and_lp_normalization(fk);
  
  concat_stats(wghsum, s0, s1, s2, stats);
  
  return 0;
}

// compute from statistics
template<class T>
int
fisher<T>::compute( T *stats, T *fk)
{
  assert(gmm);
  
  wghsum = stats[0];
  
  assert( wghsum>0 );
  
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();

  T *p=fk;

  int s1_idx = 0;
  int s2_idx = 0;
  
  // Gradient w.r.t. the mixing weights
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
      p[j] = stats[1+j] / ( wghsum*(T)sqrtf((float)iwgh[j]) );
    } 
    p += ngauss;
  }
  
  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *mean_j = gmm->mean[j];
      T *istd_j = istd+j*ndim;
      T *p_j = p+j*ndim;
      T mc = (T)sqrtf((float)iwgh[j])/wghsum;
      
      s1_idx = 1 + ngauss + j*ndim;      
      for( int k=ndim; k--; ) 
      {
        p_j[k] = mc * ( stats[s1_idx + k] - mean_j[k] * stats[1+j] ) * istd_j[k];
      }      
    }
    p += ngauss*ndim;     
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {
    int s2_start_idx = 1 + ngauss + ngauss*ndim;
    
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      s1_idx = 1 + ngauss + j*ndim;
      s2_idx = s2_start_idx + j*ndim;;
      
      T *mean_j = gmm->mean[j];
      T *var_j = gmm->var[j];
      T *p_j = p+j*ndim;
      T vc = (T)sqrtf(0.5f*(float)iwgh[j])/wghsum;
      
      for( int k=ndim; k--; ) 
      {
        p_j[k] = vc * ( ( stats[s2_idx+k] + mean_j[k] * ( mean_j[k]*stats[1+j] - (T)2.0*stats[s1_idx+k] ) ) / var_j[k] - stats[1+j] );
      }   
    }
  } 
  
  alpha_and_lp_normalization(fk);
  
  return 0;
}

template<class T>
int 
fisher<T>::compute( std::vector<T*> &x, std::vector<T> &wghx, T *fk )
{  

  assert(gmm);

  assert( x.size()==wghx.size() );

  int nsamples = (int)wghx.size();

  T wghsum=0.0;
#pragma omp parallel for reduction(+:wghsum)
  for( int i=0; i<nsamples; ++i ) 
  {
    wghsum += wghx[i];
  }

  assert( wghsum>0 );

  // accumulate statistics
  /*gmm->reset_stat_acc();
  for( int i=0; i<nsamples; ++i ) 
  {
    gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances );
  }*/
  
  ///T *s0, **s1, **s2;
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();
  {
    s0 = new T[ngauss];
    memset( s0, 0, ngauss*sizeof(T));
    s1 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s1[k] = new T[ndim];
      memset( s1[k], 0, ndim*sizeof(T));
    }
    s2 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s2[k] = new T[ndim];
      memset( s2[k], 0, ndim*sizeof(T));
    }
    for( int i=0; i<nsamples; ++i )
    {
      gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances,
				  s0, s1, s2 );
    }
  }

  T *p=fk;

  // Gradient w.r.t. the mixing weights
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
      p[j] = s0[j] / ( wghsum*(T)sqrtf((float)iwgh[j]) );
    } 
    p += ngauss;
  }

  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *mean_j = gmm->mean[j];
      T *istd_j = istd+j*ndim;
      T *p_j = p+j*ndim;
      T mc = (T)sqrtf((float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = mc * ( s1_j[k] - mean_j[k] * s0[j] ) * istd_j[k];
      }      
    }
    p += ngauss*ndim;     
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {

#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *s2_j = s2[j];
      T *mean_j = gmm->mean[j];
      T *var_j = gmm->var[j];
      T *p_j = p+j*ndim;
      T vc = (T)sqrtf(0.5f*(float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = vc * ( ( s2_j[k] + mean_j[k] * ( mean_j[k]*s0[j] - (T)2.0*s1_j[k] ) ) / var_j[k] - s0[j] );
      }   
    }
  } 
  
  alpha_and_lp_normalization(fk);
  
  return 0;
}


template<class T>
int 
fisher<T>::test( std::vector<T*> &x, T *stats )
{  

  assert(gmm);

  int nsamples = x.size();

  T wghsum=1.0*nsamples;
  
  assert( wghsum>0 );

  // accumulate statistics
  /*gmm->reset_stat_acc();
  for( int i=0; i<nsamples; ++i ) 
  {
    gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances );
  }*/
  
  ///T *s0, **s1, **s2;
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();
  {
    s0 = new T[ngauss];
    memset( s0, 0, ngauss*sizeof(T));
    s1 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s1[k] = new T[ndim];
      memset( s1[k], 0, ndim*sizeof(T));
    }
    s2 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s2[k] = new T[ndim];
      memset( s2[k], 0, ndim*sizeof(T));
    }
    for( int i=0; i<nsamples; ++i )
    {
      gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances,
				  s0, s1, s2 );
    }
  }
  
    int s1_start_idx = 1 + ngauss;
    int s2_start_idx = 1 + ngauss + ngauss*ndim;
    
    T * q = stats;
    if (wghsum != stats[0])
        std::cout << "Wghsum failed wghsum = " << wghsum << ", while stats[0] = " << stats[0] << std::endl;   
    
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
        if (q[1+j] != s0[j])
            std::cout << "S0 failed at j = " << 1+j << ", q[1+j]=" << q[1+j] << ", s0[j]=" << s0[j] << ", diff=" << q[1+j] - s0[j] << std::endl;
        
        T *s1_j = s1[j]; // s1
        T *s2_j = s2[j]; // s2
        T *q1_j = q + s1_start_idx + j*ndim;
        T *q2_j = q + s2_start_idx + j*ndim;

        for( int k=ndim; k--; ) 
        {
            if (q1_j[k] != s1_j[k])
                //std::cout << "S1 failed at j = " << s1_start_idx + j*ndim + k << std::endl;
                std::cout << "S1 failed at j = " << s1_start_idx + j*ndim + k  << ", q1_j[k]=" << q1_j[k] << ", s1_j[k]=" << s1_j[k] << ", diff=" << q1_j[k] - s1_j[k] << std::endl;
            if (q2_j[k] != s2_j[k])
                //std::cout << "S2 failed at j = " << s2_start_idx + j*ndim + k << std::endl;
                std::cout << "S2 failed at j = " << s2_start_idx + j*ndim + k  << ", q2_j[k]=" << q2_j[k] << ", s2_j[k]=" << s2_j[k] << ", diff=" << q2_j[k] - s2_j[k] << std::endl;
        }      
    }
    
  return 0;
}

template<class T>
int fisher<T>::concat_stats( T w, T* s0, T** s1, T** s2, T *stats)
{
    int ngauss = gmm->n_gauss();
    int ndim = gmm->n_dim();
  
    /* memcpy for float, using std::copy for copying between different types */
    memcpy(stats, &w, sizeof(T));
    
    memcpy(1+stats, s0, ngauss*sizeof(T));
    
    T * p = 1 + stats + ngauss;     
    T * q = 1 + stats + ngauss + ngauss*ndim;     
    
    for( int j=ngauss; j--; )
    {
        memcpy(p + j*ndim, s1[j], ndim*sizeof(T));
        memcpy(q + j*ndim, s2[j], ndim*sizeof(T));
    }
    
    if (0)
    {
        for( int j=0; j<ngauss; j++ ) 
        {
            // check w
            if (stats[0] != w)
                std::cout << "***concat_stats: w failed" << ", stats[0]=" << stats[0] << ", w=" << w << ", diff = " << stats[0] - w << std::endl;    
            
            // check s_0
            if (stats[1+j] != s0[j])
               std::cout << "***concat_stats: S0 failed at j = " << j << std::endl;
                    
            // check s_1, s_2
            for( int k=ndim; k--; ) 
            {
                if (stats[1+ngauss+j*ndim+k] != s1[j][k])
                   std::cout << "***concat_stats: S1 failed at j = " << j << std::endl;
                
                if (stats[1+ngauss+ngauss*ndim+j*ndim+k] != s2[j][k])
                   std::cout << "***concat_stats: S2 failed at j = " << j << std::endl;
            }    
        }
    }
    
    return 0;    
}

template<class T>
int 
fisher<T>::compute( std::vector<T*> &x, std::vector<T> &wghx, T *fk, T *stats)
{  

  assert(gmm);

  assert( x.size()==wghx.size() );

  int nsamples = (int)wghx.size();

  T wghsum=0.0;
#pragma omp parallel for reduction(+:wghsum)
  for( int i=0; i<nsamples; ++i ) 
  {
    wghsum += wghx[i];
  }

  assert( wghsum>0 );

  // accumulate statistics
  /*gmm->reset_stat_acc();
  for( int i=0; i<nsamples; ++i ) 
  {
    gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances );
  }*/
  
  ///T *s0, **s1, **s2;
  int ngauss = gmm->n_gauss();
  int ndim = gmm->n_dim();
  {
    s0 = new T[ngauss];
    memset( s0, 0, ngauss*sizeof(T));
    s1 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s1[k] = new T[ndim];
      memset( s1[k], 0, ndim*sizeof(T));
    }
    s2 = new T*[ngauss];
    for( int k=ngauss; k--; )
    {
      s2[k] = new T[ndim];
      memset( s2[k], 0, ndim*sizeof(T));
    }
    for( int i=0; i<nsamples; ++i )
    {
      gmm->accumulate_statistics( x[i], true, param.grad_means||param.grad_variances, param.grad_variances,
				  s0, s1, s2 );
    }
  }

  T *p=fk;
  
  // Gradient w.r.t. the mixing weights
  // without the constraint \sum_i pi_i=1 => Soft-BoV
  if( param.grad_weights )
  {
    for( int j=ngauss; j--; ) 
    {        
      p[j] = s0[j] / ( wghsum*(T)sqrtf((float)iwgh[j]) );
    } 
    p += ngauss;
  }
  
  // Gradient w.r.t. the means
  if( param.grad_means )
  {
#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *mean_j = gmm->mean[j];
      T *istd_j = istd+j*ndim;
      T *p_j = p+j*ndim;
      T mc = (T)sqrtf((float)iwgh[j])/wghsum;
      
      for( int k=ndim; k--; ) 
      {
        p_j[k] = mc * ( s1_j[k] - mean_j[k] * s0[j] ) * istd_j[k];
      }      
    }
    p += ngauss*ndim;     
  }

  // Gradient w.r.t. the variances
  if( param.grad_variances )
  {

#pragma omp parallel for
    for( int j=0; j<ngauss; j++ ) 
    {
      T *s1_j = s1[j];
      T *s2_j = s2[j];
      T *mean_j = gmm->mean[j];
      T *var_j = gmm->var[j];
      T *p_j = p+j*ndim;
      T vc = (T)sqrtf(0.5f*(float)iwgh[j])/wghsum;

      for( int k=ndim; k--; ) 
      {
        p_j[k] = vc * ( ( s2_j[k] + mean_j[k] * ( mean_j[k]*s0[j] - (T)2.0*s1_j[k] ) ) / var_j[k] - s0[j] );
      }   
    }
  } 
  
  alpha_and_lp_normalization(fk);
  
  concat_stats(wghsum, s0, s1, s2, stats);

  return 0;
}

template<class T>
void
fisher<T>::alpha_and_lp_normalization( T *fk )
{
  // alpha normalization
  if( !equal(param.alpha,1.0f) )
  {
    if( equal(param.alpha,0.5f) )
    {
#pragma omp parallel for
      for( int i=0; i<fkdim; i++ )
      {
        if( fk[i]<0.0 )
          fk[i] = -std::sqrt(-fk[i]);
        else
          fk[i] = std::sqrt(fk[i]);
      }
    }
    else
    {
#pragma omp parallel for
      for( int i=0; i<fkdim; i++ )
      {
        if( fk[i]<0.0 )
          fk[i] = -std::pow(-fk[i],(T)param.alpha);
        else
          fk[i] = std::pow(fk[i],(T)param.alpha);
      }
    }
  }

  // Lp normalization
  if( !equal(param.pnorm,(float)0.0) )
  {
    T pnorm=0;
    if( equal(param.pnorm,(float)1.0) )
    {
#pragma omp parallel for reduction(+:pnorm)
      for( int i=0; i<fkdim; ++i )
      {
        pnorm += std::fabs(fk[i]);
      }
    }
    else if( equal(param.pnorm,2.0) )
    {
      pnorm = sqrt( simd::dot_product( fkdim, fk, fk ) );
    }
    else
    {
#pragma omp parallel for reduction(+:pnorm)
      for( int i=0; i<fkdim; ++i )
      {
        pnorm += std::pow( std::fabs(fk[i]), (T)param.pnorm );
      }
      pnorm = std::pow((double) pnorm, (double)1.0/(T)param.pnorm );
    }

    if( pnorm>0.0 )
    {
      simd::scale( fkdim, fk, (T)(1.0/pnorm) );
    }
  }
}

/// \bief print
/// 
/// \param none
///
/// \return none
///
/// \author Jorge Sanchez
/// \date    August 2009

void
fisher_param::print()
{
  std::cout << "  grad_weights = " << grad_weights << std::endl;
  std::cout << "  grad_means = " << grad_means << std::endl;
  std::cout << "  grad_variances = " << grad_variances << std::endl;
  std::cout << "  alpha = " << alpha << std::endl;
  std::cout << "  pnorm = " << pnorm << std::endl;
}

template class fisher<float>;
template class fisher<double>;
